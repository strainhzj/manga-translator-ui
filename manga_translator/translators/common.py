import re
import time
import asyncio
from typing import List, Tuple
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper, repeating_sequence, is_valuable_text

try:
    import readline
except Exception:
    readline = None

VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'POL': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
    'ARA': 'Arabic',
    'CNR': 'Montenegrin',
    'SRP': 'Serbian',
    'HRV': 'Croatian',
    'THA': 'Thai',
    'IND': 'Indonesian',
    'FIL': 'Filipino (Tagalog)'
}

ISO_639_1_TO_VALID_LANGUAGES = {
    'zh': 'CHS',
    'ja': 'JPN',
    'en': 'ENG',
    'ko': 'KOR',
    'vi': 'VIN',
    'cs': 'CSY',
    'nl': 'NLD',
    'fr': 'FRA',
    'de': 'DEU',
    'hu': 'HUN',
    'it': 'ITA',
    'pl': 'POL',
    'pt': 'PTB',
    'ro': 'ROM',
    'ru': 'RUS',
    'es': 'ESP',
    'tr': 'TRK',
    'uk': 'UKR',
    'vi': 'VIN',
    'ar': 'ARA',
    'cnr': 'CNR',
    'sr': 'SRP',
    'hr': 'HRV',
    'th': 'THA',
    'id': 'IND',
    'tl': 'FIL'
}

class InvalidServerResponse(Exception):
    pass

class MissingAPIKeyException(Exception):
    pass

class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error)

class BRMarkersValidationException(Exception):
    """AI断句检查失败异常"""
    def __init__(self, missing_count: int, total_count: int, tolerance: int):
        self.missing_count = missing_count
        self.total_count = total_count
        self.tolerance = tolerance
        super().__init__(
            f"AI断句检查失败：{missing_count}/{total_count} 条翻译缺失[BR]标记（容忍度：{tolerance}）"
        )

class MultimodalUnsupportedException(Exception):
    """模型不支持多模态输入异常"""
    def __init__(self, model_name: str, translator: str):
        self.model_name = model_name
        self.translator = translator
        super().__init__(
            f"模型 {model_name} 不支持多模态输入（图片+文本）"
        )

class MTPEAdapter():
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        # TODO: Make it work in windows (e.g. through os.startfile)
        if not readline:
            print('MTPE is currently only supported on linux')
            return translations
        new_translations = []
        print('Running Machine Translation Post Editing (MTPE)')
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f'\n[{i + 1}/{len(queries)}] {query}:')
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace('\n', '\\n')))
            new_translation = ''
            try:
                new_translation = input(' -> ').replace('\\n', '\n')
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        print()
        return new_translations

class CommonTranslator(InfererModule):
    # Translator has to support all languages listed in here. The language codes will be resolved into
    # _LANGUAGE_CODE_MAP[lang_code] automatically if _LANGUAGE_CODE_MAP is a dict.
    # If it is a list it will simply return the language code as is.
    _LANGUAGE_CODE_MAP = {}

    # The amount of repeats upon detecting an invalid translation.
    # Use with _is_translation_invalid and _modify_invalid_translation_query.
    _INVALID_REPEAT_COUNT = 0

    # Will sleep for the rest of the minute if the request count is over this number.
    _MAX_REQUESTS_PER_MINUTE = -1

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()
        self._last_request_ts = 0
        self.enable_post_translation_check = False
        self.post_check_repetition_threshold = 5
        self.post_check_max_retry_attempts = 2
        self.attempts = -1
        self._MAX_SPLIT_ATTEMPTS = 3  # 最大分割层级
        self._SPLIT_THRESHOLD = 2  # 重试N次后触发分割
        self._global_attempt_count = 0  # 全局尝试计数器
        self._max_total_attempts = -1  # 全局最大尝试次数
        self._cancel_check_callback = None  # 取消检查回调
    
    def set_cancel_check_callback(self, callback):
        """设置取消检查回调"""
        self._cancel_check_callback = callback
    
    def _check_cancelled(self):
        """检查任务是否被取消"""
        if self._cancel_check_callback and self._cancel_check_callback():
            raise asyncio.CancelledError("Translation cancelled by user")

    def _get_retry_hint(self, attempt: int, reason: str = "") -> str:
        """
        生成重试提示信息，用于避免模型服务器缓存导致的重复错误
        
        Args:
            attempt: 当前尝试次数
            reason: 重试原因（可选）
            
        Returns:
            重试提示字符串
        """
        hints = [
            f"[Retry attempt #{attempt}]",
            f"[This is attempt #{attempt}, please provide a different response]",
            f"[Attempt {attempt}: Previous response had issues, please try again]",
            f"[Retry #{attempt}: Please ensure quality this time]",
            f"[Attempt {attempt} - Previous attempts failed quality check]"
        ]
        
        # 根据尝试次数选择不同的提示（循环使用）
        base_hint = hints[(attempt - 1) % len(hints)]
        
        # 如果提供了原因，添加到提示中
        if reason:
            return f"{base_hint} Reason: {reason}\n\n"
        else:
            return f"{base_hint}\n\n"
    
    def _build_user_prompt_for_texts(self, texts: List[str], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """
        统一的用户提示词构建方法（纯文本翻译）
        适用于 openai.py 和 gemini.py

        Args:
            texts: 要翻译的文本列表
            ctx: 上下文对象（可选）
            prev_context: 历史上下文（可选）
            retry_attempt: 重试次数（可选，用于避免缓存）
            retry_reason: 重试原因（可选）

        Returns:
            构建好的用户提示词字符串
        """
        # 检查是否开启AI断句
        enable_ai_break = False
        if ctx and hasattr(ctx, 'config') and ctx.config and hasattr(ctx.config, 'render'):
            enable_ai_break = getattr(ctx.config.render, 'disable_auto_wrap', False)

        prompt = ""
        
        # 添加重试提示（如果是重试）
        if retry_attempt > 0:
            prompt += self._get_retry_hint(retry_attempt, retry_reason)

        # 添加多页上下文（如果有）
        if prev_context:
            prompt += f"{prev_context}\n\n---\n\n"
            self.logger.info(f"[历史上下文] 长度: {len(prev_context)} 字符")
            self.logger.info(f"[历史上下文内容]\n{prev_context[:500]}...")
        else:
            self.logger.info(f"[历史上下文] 无历史上下文（可能是第一张图片或context_size=0）")

        prompt += "Please translate the following manga text regions:\n\n"

        for i, text in enumerate(texts):
            text_to_translate = text.replace('\n', ' ').replace('\ufffd', '')
            # 只有开启AI断句时才添加区域信息
            if enable_ai_break and ctx and hasattr(ctx, 'text_regions') and ctx.text_regions and i < len(ctx.text_regions):
                region = ctx.text_regions[i]
                region_count = len(region.lines) if hasattr(region, 'lines') else 1
                prompt += f"{i+1}. [Original regions: {region_count}] {text_to_translate}\n"
            else:
                prompt += f"{i+1}. {text_to_translate}\n"

        prompt += "\nCRITICAL: Provide translations in the exact same order as the numbered input text regions. Your first line of output must be the translation for text region #1, your second line for #2, and so on. DO NOT CHANGE THE ORDER."

        return prompt

    def _build_user_prompt_for_hq(self, batch_data: List, ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """
        统一的用户提示词构建方法（高质量多模态翻译）
        适用于 openai_hq.py 和 gemini_hq.py

        Args:
            batch_data: 批次数据列表，包含图片和原文
            ctx: 上下文对象（可选）
            prev_context: 历史上下文（可选）
            retry_attempt: 重试次数（可选，用于避免缓存）
            retry_reason: 重试原因（可选）

        Returns:
            构建好的用户提示词字符串
        """
        # 检查是否开启AI断句
        enable_ai_break = False
        if ctx and hasattr(ctx, 'config') and ctx.config and hasattr(ctx.config, 'render'):
            enable_ai_break = getattr(ctx.config.render, 'disable_auto_wrap', False)

        prompt = ""
        
        # 添加重试提示（如果是重试）
        if retry_attempt > 0:
            prompt += self._get_retry_hint(retry_attempt, retry_reason)

        # 添加多页上下文（如果有）
        if prev_context:
            prompt += f"{prev_context}\n\n---\n\n"
            self.logger.info(f"[HQ历史上下文] 长度: {len(prev_context)} 字符")
            self.logger.info(f"[HQ历史上下文内容]\n{prev_context[:500]}...")
        else:
            self.logger.info(f"[HQ历史上下文] 无历史上下文（可能是第一张图片或context_size=0）")

        prompt += "Please translate the following manga text regions. I'm providing multiple images with their text regions in reading order:\n\n"

        # 添加图片信息
        for i, data in enumerate(batch_data):
            prompt += f"=== Image {i+1} ===\n"
            prompt += f"Text regions ({len(data['original_texts'])} regions):\n"
            for j, text in enumerate(data['original_texts']):
                prompt += f"  {j+1}. {text}\n"
            prompt += "\n"

        prompt += "All texts to translate (in order):\n"
        text_index = 1
        for img_idx, data in enumerate(batch_data):
            for region_idx, text in enumerate(data['original_texts']):
                text_to_translate = text.replace('\n', ' ').replace('\ufffd', '')
                # 只有开启AI断句时才添加区域信息
                if enable_ai_break and data.get('text_regions') and region_idx < len(data['text_regions']):
                    region = data['text_regions'][region_idx]
                    region_count = len(region.lines) if hasattr(region, 'lines') else 1
                    prompt += f"{text_index}. [Original regions: {region_count}] {text_to_translate}\n"
                else:
                    prompt += f"{text_index}. {text_to_translate}\n"
                text_index += 1

        prompt += "\nCRITICAL: Provide translations in the exact same order as the numbered input text regions. Your first line of output must be the translation for text region #1, your second line for #2, and so on. DO NOT CHANGE THE ORDER."

        return prompt

    def _validate_br_markers(self, translations: List[str], queries: List[str] = None, ctx=None, batch_indices: List[int] = None, batch_data: List = None, split_level: int = 0) -> bool:
        """
        检查翻译结果是否包含必要的[BR]标记
        Check if translations contain necessary [BR] markers
        
        Args:
            translations: 翻译结果列表
            queries: 原始查询列表（可选）
            ctx: 上下文（用于获取配置和区域信息）
            batch_indices: 批次索引列表（可选，用于定位text_regions）
            batch_data: 批次数据列表（可选，HQ翻译器使用）
            split_level: 分割级别（可选，用于跳过深度分割时的检查）
            
        Returns:
            True if validation passes, False if BR markers are missing
        """
        import re
        
        # 如果分割级别过深（>=3），跳过BR检查以避免无限重试
        if split_level >= 3:
            self.logger.info(f"[AI断句检查] 分割级别过深 (split_level={split_level})，跳过BR标记检查")
            return True
        
        # 检查是否启用了BR检查
        check_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            check_enabled = getattr(ctx.config.render, 'check_br_and_retry', False)
        
        if not check_enabled:
            return True  # 检查未启用，直接通过
        
        # 检查是否启用了AI断句
        ai_break_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            ai_break_enabled = getattr(ctx.config.render, 'disable_auto_wrap', False)
        
        if not ai_break_enabled:
            return True  # AI断句未启用，不需要检查BR
        
        # 提取每个翻译对应的区域数
        region_counts = []
        if ctx and hasattr(ctx, 'text_regions') and ctx.text_regions:
            for idx in range(len(translations)):
                # 确定实际的region索引
                if batch_indices and idx < len(batch_indices):
                    region_idx = batch_indices[idx]
                else:
                    region_idx = idx
                
                if region_idx < len(ctx.text_regions):
                    region = ctx.text_regions[region_idx]
                    region_count = len(region.lines) if hasattr(region, 'lines') else 1
                    region_counts.append(region_count)
                else:
                    region_counts.append(1)  # 默认为1
        elif batch_data:
            # HQ翻译器使用batch_data
            for idx in range(len(translations)):
                region_idx = idx
                for data in batch_data:
                    if 'text_regions' in data and data['text_regions'] and region_idx < len(data['text_regions']):
                        region = data['text_regions'][region_idx]
                        region_count = len(region.lines) if hasattr(region, 'lines') else 1
                        region_counts.append(region_count)
                        break
                else:
                    region_counts.append(1)
        else:
            region_counts = [1] * len(translations)  # 默认都为1
        
        # 检查每个翻译，统计缺失BR的数量
        needs_check_count = 0
        missing_br_count = 0
        missing_indices = []
        
        for idx, (translation, region_count) in enumerate(zip(translations, region_counts)):
            # 只检查区域数≥2的翻译
            if region_count >= 2:
                needs_check_count += 1
                # 检查是否包含BR标记
                has_br = bool(re.search(r'(\[BR\]|【BR】|<br>)', translation, flags=re.IGNORECASE))
                if not has_br:
                    missing_br_count += 1
                    missing_indices.append(idx + 1)
                    self.logger.warning(
                        f"Translation {idx+1} missing [BR] markers (expected for {region_count} regions): {translation[:50]}..."
                    )
        
        # 计算容忍的错误数量：十分之一，最少1个
        if needs_check_count > 0:
            tolerance = max(1, needs_check_count // 10)
            
            if missing_br_count > tolerance:
                # 超过容忍度，验证失败
                self.logger.warning(
                    f"[AI断句检查] 缺失BR标记的翻译数 ({missing_br_count}/{needs_check_count}) 超过容忍度 ({tolerance})，需要重试"
                )
                return False
            elif missing_br_count > 0:
                # 在容忍度内，警告但通过
                self.logger.warning(
                    f"[AI断句检查] ⚠ {missing_br_count}/{needs_check_count} 条翻译缺失BR标记，但在容忍度内 ({tolerance})，继续执行"
                )
                return True
            else:
                # 全部通过
                self.logger.info(f"[AI断句检查] ✓ 所有多行区域的翻译都包含[BR]标记 (检查了 {needs_check_count}/{len(translations)} 条)")
                return True

        return True  # 没有需要检查的翻译，直接通过

    def _validate_translation_quality(self, queries: List[str], translations: List[str]) -> Tuple[bool, str]:
        """
        验证翻译质量，检测常见问题

        Args:
            queries: 原文列表
            translations: 译文列表

        Returns:
            (is_valid, error_message)
        """
        import string

        # 1. 检查数量匹配
        if len(translations) != len(queries):
            return False, f"Translation count mismatch: expected {len(queries)}, got {len(translations)}"

        # 2. 检查空翻译（原文不为空但译文为空）
        empty_translation_errors = []
        for i, (source, translation) in enumerate(zip(queries, translations)):
            if source.strip() and not translation.strip():
                empty_translation_errors.append(i + 1)

        if empty_translation_errors:
            return False, f"Empty translation detected at positions: {empty_translation_errors}"

        # 3. 检查合并翻译（原文是正常文本但译文只有标点）
        for i, (source, translation) in enumerate(zip(queries, translations)):
            is_source_simple = all(char in string.punctuation or char.isspace() for char in source)
            is_translation_simple = all(char in string.punctuation or char.isspace() for char in translation)

            if is_translation_simple and not is_source_simple:
                return False, f"Detected potential merged translation at position {i+1}"

        # 4. 检查可疑符号（模型幻觉）
        SUSPICIOUS_SYMBOLS = ["ହ", "ି", "ഹ"]
        for symbol in SUSPICIOUS_SYMBOLS:
            for translation in translations:
                if symbol in translation:
                    return False, f"Suspicious symbol '{symbol}' detected in translation"

        return True, ""

    def _reset_global_attempt_count(self):
        """重置全局尝试计数器（每次新的翻译任务开始时调用）"""
        self._global_attempt_count = 0
        self._max_total_attempts = self.attempts

    def _increment_global_attempt(self) -> bool:
        """
        增加全局尝试计数，返回是否还可以继续尝试

        Returns:
            True: 还可以继续尝试
            False: 已达到总次数上限
        """
        self._global_attempt_count += 1

        # 无限重试模式
        if self._max_total_attempts == -1:
            return True

        # 检查是否超过上限
        if self._global_attempt_count >= self._max_total_attempts:
            self.logger.warning(f"Reached max total attempts: {self._global_attempt_count}/{self._max_total_attempts}")
            return False

        return True

    class SplitException(Exception):
        """用于触发分割的特殊异常"""
        def __init__(self, attempt_count, texts):
            self.attempt_count = attempt_count
            self.texts = texts
            super().__init__(f"Split triggered after {attempt_count} attempts")

    async def _translate_with_split(self, translator_func, texts: List[str], split_level: int = 0, **kwargs) -> List[str]:
        """
        带分割重试的翻译包装器（新逻辑）

        Args:
            translator_func: 实际的翻译函数（async callable）
            texts: 要翻译的文本列表
            split_level: 当前分割层级
            **kwargs: 传递给translator_func的其他参数

        Returns:
            翻译结果列表
        """
        # 检查是否超过全局尝试次数
        if self._max_total_attempts != -1 and self._global_attempt_count >= self._max_total_attempts:
            self.logger.error(f"Global attempt limit reached before translation: {self._global_attempt_count}/{self._max_total_attempts}")
            raise Exception(f"Translation failed: reached max total attempts ({self._max_total_attempts})")

        try:
            # 尝试翻译（内部会检查是否需要分割）
            translations = await translator_func(texts, split_level=split_level, **kwargs)
            return translations

        except self.SplitException as split_ex:
            # 触发分割
            if split_level < self._MAX_SPLIT_ATTEMPTS and len(texts) > 1:
                self.logger.warning(
                    f"Splitting after {split_ex.attempt_count} attempts at split_level={split_level}, "
                    f"batch size {len(texts)} → splitting into two halves"
                )

                # 分成两半（只分割texts，不分割batch_data等其他参数）
                mid = len(texts) // 2
                left_texts = texts[:mid]
                right_texts = texts[mid:]

                self.logger.info(f"Split: left={len(left_texts)}, right={len(right_texts)}, global_attempts={self._global_attempt_count}/{self._max_total_attempts}")

                # 并发翻译左右两部分（kwargs保持完整传递）
                try:
                    left_translations, right_translations = await asyncio.gather(
                        self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs),
                        self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs),
                        return_exceptions=False
                    )
                except Exception as split_error:
                    # 如果并发失败，回退到串行处理
                    self.logger.warning(f"Concurrent split failed, falling back to sequential: {split_error}")
                    left_translations = await self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs)
                    right_translations = await self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs)

                # 合并结果
                return left_translations + right_translations

            else:
                # 不能再分割了，终止翻译进程
                if len(texts) == 1:
                    self.logger.error(f"Single text translation failed at split_level={split_level}: {texts[0][:50]}...")
                    raise Exception(f"Translation failed for single text after {split_ex.attempt_count} attempts")
                else:
                    self.logger.error(f"Max split level ({self._MAX_SPLIT_ATTEMPTS}) reached, batch size={len(texts)}")
                    raise Exception(f"Translation failed: max split level reached with batch size {len(texts)}")

        except Exception as e:
            # 其他异常（非分割触发的），直接终止
            self.logger.error(f"Translation failed with exception at split_level={split_level}: {e}")
            raise e

    def parse_args(self, config):
        self.enable_post_translation_check = getattr(config, 'enable_post_translation_check', self.enable_post_translation_check)
        self.post_check_repetition_threshold = getattr(config, 'post_check_repetition_threshold', self.post_check_repetition_threshold)
        self.post_check_max_retry_attempts = getattr(config, 'post_check_max_retry_attempts', self.post_check_max_retry_attempts)
        self.attempts = getattr(config, 'attempts', self.attempts)

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ['auto'] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None
        if type(self._LANGUAGE_CODE_MAP) is list:
            return from_lang, to_lang

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False, ctx=None) -> List[str]:
        """
        Translates list of queries of one language into another.
        """
        if to_lang not in VALID_LANGUAGES:
            raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (to_lang, ', '.join(VALID_LANGUAGES)))
        if from_lang not in VALID_LANGUAGES and from_lang != 'auto':
            raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (from_lang, ', '.join(VALID_LANGUAGES)))
        self.logger.info(f'Translating into {VALID_LANGUAGES[to_lang]}')

        if from_lang == to_lang:
            # 即使源语言和目标语言相同，也应用文本清理（如全角句点替换）
            return [self._clean_translation_output(q, q, to_lang) for q in queries]

        # Dont translate queries without text
        query_indices = []
        final_translations = []
        for i, query in enumerate(queries):
            if not is_valuable_text(query):
                final_translations.append(queries[i])
            else:
                final_translations.append(None)
                query_indices.append(i)

        queries = [queries[i] for i in query_indices]

        translations = [''] * len(queries)
        untranslated_indices = list(range(len(queries)))
        for i in range(1 + self._INVALID_REPEAT_COUNT): # Repeat until all translations are considered valid
            if i > 0:
                self.logger.warn(f'Repeating because of invalid translation. Attempt: {i+1}')
                await asyncio.sleep(0.1)

            # Sleep if speed is over the ratelimit
            await self._ratelimit_sleep()

            # Translate
            _translations = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries, ctx=ctx)

            # Strict validation: translation count must match query count
            if len(_translations) != len(queries):
                error_msg = f"Translation count mismatch: expected {len(queries)}, got {len(_translations)}"
                self.logger.error(error_msg)
                self.logger.error(f"Queries: {queries}")
                self.logger.error(f"Translations: {_translations}")
                raise InvalidServerResponse(error_msg)

            # Only overwrite yet untranslated indices
            for j in untranslated_indices:
                translations[j] = _translations[j]

            if self._INVALID_REPEAT_COUNT == 0:
                break

            new_untranslated_indices = []
            for j in untranslated_indices:
                q, t = queries[j], translations[j]
                # Repeat invalid translations with slightly modified queries
                if self._is_translation_invalid(q, t):
                    new_untranslated_indices.append(j)
                    queries[j] = self._modify_invalid_translation_query(q, t)
            untranslated_indices = new_untranslated_indices

            if not untranslated_indices:
                break

        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]

        if to_lang == 'ARA':
            import arabic_reshaper , bidi.algorithm
            translations = [bidi.algorithm.get_display(arabic_reshaper.reshape(t)) for t in translations]

        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)

        # Merge with the queries without text
        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f'{i}: {queries[i]} => {trans}')

        return final_translations

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        pass

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            ratelimit_timeout = self._last_request_ts + 60 / self._MAX_REQUESTS_PER_MINUTE
            if ratelimit_timeout > now:
                self.logger.info(f'Ratelimit sleep: {(ratelimit_timeout-now):.2f}s')
                await asyncio.sleep(ratelimit_timeout-now)
            self._last_request_ts = time.time()

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        if not trans and query:
            return True
        if not query or not trans:
            return False

        query_symbols_count = len(set(query))
        trans_symbols_count = len(set(trans))
        if query_symbols_count > 6 and trans_symbols_count < 6 and trans_symbols_count < 0.25 * len(trans):
            return True
        return False

    def _modify_invalid_translation_query(self, query: str, trans: str) -> str:
        """
        Can be overwritten if _INVALID_REPEAT_COUNT was set. It modifies the query
        for the next translation attempt.
        """
        return query

    def _clean_translation_output(self, query: str, trans: str, to_lang: str) -> str:
        """
        Tries to spot and skim down invalid translations.
        """
        if not query or not trans:
            return ''

        # 移除内部标记：【Original regions: X】或 [Original regions: X]
        # Remove internal markers: 【Original regions: X】 or [Original regions: X]
        trans = re.sub(r'【Original regions:\s*\d+】\s*', '', trans, flags=re.IGNORECASE)
        trans = re.sub(r'\[Original regions:\s*\d+\]\s*', '', trans, flags=re.IGNORECASE)
        
        # 替换全角句点连续出现（．．．或．．）为省略号
        trans = trans.replace('．．．', '…')
        trans = trans.replace('．．', '…')

        # '  ' -> ' '
        trans = re.sub(r'\s+', r' ', trans)
        # 'text.text' -> 'text. text'
        trans = re.sub(r'(?<![.,;!?])([.,;!?])(?=\w)', r'\1 ', trans)
        # ' ! ! . . ' -> ' !!.. '
        trans = re.sub(r'([.,;!?])\s+(?=[.,;!?]|$)', r'\1', trans)

        if to_lang != 'ARA':
            # 'text .' -> 'text.'
            trans = re.sub(r'(?<=[.,;!?\w])\s+([.,;!?])', r'\1', trans)
            # ' ... text' -> ' ...text'
            trans = re.sub(r'((?:\s|^)\.+)\s+(?=\w)', r'\1', trans)

        seq = repeating_sequence(trans.lower())

        # 'aaaaaaaaaaaaa' -> 'aaaaaa'
        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            # Shrink sequence to length of original query
            trans = seq * max(len(query) // len(seq), 1)
            # Transfer capitalization of query to translation
            nTrans = ''
            for i in range(min(len(trans), len(query))):
                nTrans += trans[i].upper() if query[i].isupper() else trans[i]
            trans = nTrans

        # words = text.split()
        # elements = list(set(words))
        # if len(elements) / len(words) < 0.1:
        #     words = words[:int(len(words) / 1.75)]
        #     text = ' '.join(words)

        #     # For words that appear more then four times consecutively, remove the excess
        #     for el in elements:
        #         el = re.escape(el)
        #         text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return trans

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_SUB_DIR = 'translators'

    async def _translate(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)
    
    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def unload(self, device: str):
        return await super().unload()
