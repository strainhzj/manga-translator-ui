# -*- coding: utf-8 -*-
"""
依赖包检查工具
Package checking utilities
"""

import functools
import itertools
import pathlib
import subprocess
import sys
from typing import List, Optional

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import importlib.metadata as importlib_metadata
except (ModuleNotFoundError, ImportError):
    import importlib_metadata
from packaging.version import Version


def package_version(name: str) -> Optional[Version]:
    """获取已安装包的版本"""
    try:
        return Version(importlib_metadata.distribution(canonicalize_name(name)).version)
    except importlib_metadata.PackageNotFoundError:
        return None


def _nonblank(text):
    return text and not text.startswith('#')


@functools.singledispatch
def yield_lines(iterable):
    """提取有效行"""
    return itertools.chain.from_iterable(map(yield_lines, iterable))


@yield_lines.register(str)
def _(text):
    return filter(_nonblank, map(str.strip, text.splitlines()))


def drop_comment(line):
    """去除注释"""
    return line.partition(' #')[0]


def join_continuation(lines):
    """合并续行"""
    lines = iter(lines)
    for item in lines:
        while item.endswith('\\'):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item


def load_req_file(requirements_file: str) -> List[str]:
    """加载requirements文件"""
    with pathlib.Path(requirements_file).open() as reqfile:
        return list(map(
            lambda x: str(Requirement(x)),
            join_continuation(map(drop_comment, yield_lines(reqfile)))
        ))


def _yield_reqs_to_install(req: Requirement, current_extra: str = ''):
    """检查需要安装的依赖"""
    if req.marker and not req.marker.evaluate({'extra': current_extra}):
        return

    try:
        version = importlib_metadata.distribution(req.name).version
    except importlib_metadata.PackageNotFoundError:
        yield req
    else:
        if req.specifier.contains(version, prereleases=True):
            for child_req in (importlib_metadata.metadata(req.name).get_all('Requires-Dist') or []):
                child_req_obj = Requirement(child_req)
                need_check, ext = False, None
                for extra in req.extras:
                    if child_req_obj.marker and child_req_obj.marker.evaluate({'extra': extra}):
                        need_check = True
                        ext = extra
                        break
                if need_check:
                    yield from _yield_reqs_to_install(child_req_obj, ext)
        else:
            yield req


def _check_req(req: Requirement):
    """检查单个依赖是否满足"""
    return not bool(list(itertools.islice(_yield_reqs_to_install(req), 1)))


def check_reqs(reqs: List[str]) -> bool:
    """检查所有依赖是否满足"""
    return all(map(lambda x: _check_req(Requirement(x)), reqs))


def check_req_file(requirements_file: str) -> bool:
    """检查requirements文件中的依赖是否满足"""
    try:
        return check_reqs(load_req_file(requirements_file))
    except Exception as e:
        print(f'检查依赖文件失败: {e}')
        return False

