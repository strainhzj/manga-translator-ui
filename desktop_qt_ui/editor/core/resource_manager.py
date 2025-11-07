"""资源管理器

统一管理编辑器的所有资源，包括图片、蒙版、区域等。
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .resources import ImageResource, MaskResource, RegionResource
from .types import MaskType


class ResourceManager:
    """资源管理器
    
    统一管理所有编辑器资源的生命周期。
    """
    
    def __init__(self):
        """初始化资源管理器"""
        self.logger = logging.getLogger(__name__)
        
        # 当前加载的资源
        self._current_image: Optional[ImageResource] = None
        self._masks: Dict[MaskType, MaskResource] = {}
        self._regions: Dict[int, RegionResource] = {}
        self._next_region_id = 0
        
        # 资源缓存（用于快速切换）
        self._image_cache: Dict[str, ImageResource] = {}
        self._cache_limit = 5  # 最多缓存5张图片
        
        # 通用缓存（用于存储临时数据）
        self._temp_cache: Dict[str, any] = {}
    
    # ==================== 图片管理 ====================
    
    def load_image(self, image_path: str, json_data: Optional[Dict] = None) -> ImageResource:
        """加载图片资源
        
        Args:
            image_path: 图片路径
            json_data: 关联的JSON数据
        
        Returns:
            ImageResource: 图片资源
        
        Raises:
            FileNotFoundError: 图片文件不存在
            Exception: 加载失败
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 规范化路径
        image_path = os.path.normpath(image_path)
        
        # 检查缓存
        if image_path in self._image_cache:
            self.logger.debug(f"Image loaded from cache: {image_path}")
            resource = self._image_cache[image_path]
            self._current_image = resource
            return resource
        
        # 加载图片
        try:
            self.logger.info(f"Loading image: {image_path}")
            image = Image.open(image_path)
            
            # 创建资源对象
            resource = ImageResource(
                path=image_path,
                image=image,
                width=image.width,
                height=image.height,
                json_data=json_data,
            )
            
            # 添加到缓存
            self._add_to_cache(image_path, resource)
            
            # 设置为当前图片
            self._current_image = resource
            
            self.logger.info(f"Image loaded successfully: {image_path} ({image.width}x{image.height})")
            return resource
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _add_to_cache(self, path: str, resource: ImageResource) -> None:
        """添加图片到缓存
        
        Args:
            path: 图片路径
            resource: 图片资源
        """
        # 如果缓存已满，删除最旧的
        if len(self._image_cache) >= self._cache_limit:
            # 按加载时间排序，删除最旧的
            oldest_path = min(self._image_cache.items(), key=lambda x: x[1].load_time)[0]
            old_resource = self._image_cache.pop(oldest_path)
            old_resource.release()
            self.logger.debug(f"Removed oldest image from cache: {oldest_path}")
        
        self._image_cache[path] = resource
    
    def unload_image(self) -> None:
        """卸载当前图片及所有关联资源"""
        if self._current_image:
            self.logger.info(f"Unloading image: {self._current_image.path}")
            self._current_image = None
        
        # 清空所有关联资源
        self.clear_masks()
        self.clear_regions()
    
    def get_current_image(self) -> Optional[ImageResource]:
        """获取当前图片资源
        
        Returns:
            Optional[ImageResource]: 当前图片资源，如果没有加载返回None
        """
        return self._current_image
    
    def get_current_image_path(self) -> Optional[str]:
        """获取当前图片路径
        
        Returns:
            Optional[str]: 当前图片路径，如果没有加载返回None
        """
        return self._current_image.path if self._current_image else None
    
    # ==================== 蒙版管理 ====================
    
    def set_mask(self, mask_type: MaskType, mask_data: np.ndarray) -> MaskResource:
        """设置蒙版
        
        Args:
            mask_type: 蒙版类型
            mask_data: 蒙版数据
        
        Returns:
            MaskResource: 蒙版资源
        """
        if not self._current_image:
            raise RuntimeError("No image loaded")
        
        # 创建蒙版资源
        resource = MaskResource(
            mask_type=mask_type,
            data=mask_data.copy(),
            width=mask_data.shape[1],
            height=mask_data.shape[0],
        )
        
        # 释放旧蒙版
        if mask_type in self._masks:
            self._masks[mask_type].release()
        
        self._masks[mask_type] = resource
        self.logger.debug(f"Set mask: {mask_type}")
        return resource
    
    def get_mask(self, mask_type: MaskType) -> Optional[MaskResource]:
        """获取蒙版
        
        Args:
            mask_type: 蒙版类型
        
        Returns:
            Optional[MaskResource]: 蒙版资源，如果不存在返回None
        """
        return self._masks.get(mask_type)
    
    def clear_masks(self) -> None:
        """清空所有蒙版"""
        for mask in self._masks.values():
            mask.release()
        self._masks.clear()
        self.logger.debug("Cleared all masks")
    
    # ==================== 区域管理 ====================
    
    def add_region(self, region_data: Dict) -> RegionResource:
        """添加文本区域
        
        Args:
            region_data: 区域数据
        
        Returns:
            RegionResource: 区域资源
        """
        region_id = self._next_region_id
        self._next_region_id += 1
        
        resource = RegionResource(
            region_id=region_id,
            data=region_data.copy(),
        )
        
        self._regions[region_id] = resource
        self.logger.debug(f"Added region: {region_id}")
        return resource
    
    def update_region(self, region_id: int, updates: Dict) -> None:
        """更新区域数据
        
        Args:
            region_id: 区域ID
            updates: 要更新的数据
        
        Raises:
            KeyError: 区域不存在
        """
        if region_id not in self._regions:
            self.logger.warning(f"Region {region_id} not found, skipping update")
            return  # 静默失败，不抛出异常
        
        self._regions[region_id].data.update(updates)
        import time
        self._regions[region_id].update_time = time.time()
        self.logger.debug(f"Updated region: {region_id}")
    
    def delete_region(self, region_id: int) -> None:
        """删除区域
        
        Args:
            region_id: 区域ID
        """
        if region_id in self._regions:
            del self._regions[region_id]
            self.logger.debug(f"Deleted region: {region_id}")
    
    def get_region(self, region_id: int) -> Optional[RegionResource]:
        """获取区域
        
        Args:
            region_id: 区域ID
        
        Returns:
            Optional[RegionResource]: 区域资源，如果不存在返回None
        """
        return self._regions.get(region_id)
    
    def get_all_regions(self) -> List[RegionResource]:
        """获取所有区域
        
        Returns:
            List[RegionResource]: 区域列表
        """
        return list(self._regions.values())
    
    def clear_regions(self) -> None:
        """清空所有区域"""
        self._regions.clear()
        self._next_region_id = 0
        self.logger.debug("Cleared all regions")
    
    # ==================== 缓存管理 ====================
    
    def set_cache(self, key: str, value: any) -> None:
        """设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        self._temp_cache[key] = value
        self.logger.debug(f"Set cache: {key}")
    
    def get_cache(self, key: str, default=None) -> any:
        """获取缓存数据
        
        Args:
            key: 缓存键
            default: 默认值
        
        Returns:
            缓存值，如果不存在返回default
        """
        return self._temp_cache.get(key, default)
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """清空缓存
        
        Args:
            key: 如果指定，只清空该键；否则清空所有缓存
        """
        if key:
            if key in self._temp_cache:
                del self._temp_cache[key]
                self.logger.debug(f"Cleared cache: {key}")
        else:
            self._temp_cache.clear()
            self.logger.debug("Cleared all cache")
    
    # ==================== 资源清理 ====================
    
    def cleanup_all(self) -> None:
        """清理所有资源"""
        self.logger.info("Cleaning up all resources")
        
        # 卸载当前图片
        self.unload_image()
        
        # 清理缓存
        for resource in self._image_cache.values():
            resource.release()
        self._image_cache.clear()
        
        self.logger.info("All resources cleaned up")
    
    def get_memory_usage_estimate(self) -> int:
        """估算内存使用量（字节）
        
        Returns:
            int: 估算的内存使用量
        """
        size = 0
        
        # 图片缓存
        for resource in self._image_cache.values():
            if resource.image:
                # 估算：宽 × 高 × 通道数 × 每像素字节数
                size += resource.width * resource.height * 4  # RGBA
        
        # 蒙版
        for mask in self._masks.values():
            if mask.data is not None:
                size += mask.data.nbytes
        
        return size
    
    def __del__(self):
        """析构函数"""
        self.cleanup_all()

