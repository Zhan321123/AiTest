import os
import re
from pathlib import Path


def increasePath(filePath: Path) -> Path:
  """
  路径名递增（确保返回的路径不存在）
  :param filePath: 原始路径
  :return: 递增后的新路径（不存在）
  """
  if not filePath.parent.exists():
    print(f"父级目录 {filePath.parent} 不存在")
    return filePath  # 父目录不存在时返回原路径（无法创建有效路径）

  if not filePath.exists():
    return filePath

  name, ext = os.path.splitext(filePath.name)
  pattern = r'^(.*?)\((\d+)\)$'
  match = re.match(pattern, name)

  if match:
    prefix = match.group(1)
    current_number = int(match.group(2)) + 1
  else:
    prefix = name
    current_number = 1
  while True:
    new_name = f"{prefix}({current_number}){ext}"
    new_path = filePath.parent / new_name
    if not new_path.exists():
      return new_path
    current_number += 1


def _printClass(obj: object):
  """
  打印类继承结构

  :param obj:
  :return:
  """
  print(f"[{obj.__class__.__name__} - 继承结构]".center(50, "-"))
  try:
    mro = obj.__class__.mro()
    for i, cls in enumerate(mro):
      print(f"  {i}. {cls.__module__}.{cls.__name__}")
  except AttributeError:
    print("  无法获取继承结构")
  print(f"[{obj.__class__.__name__} - 子类信息]".center(50, "-"))
  try:
    subclasses = obj.__class__.__subclasses__()
    if subclasses:
      for i, subcls in enumerate(subclasses):
        print(f"  {i}. {subcls.__module__}.{subcls.__name__}")
    else:
      print("  该类没有子类")
  except AttributeError:
    print("  无法获取子类信息")


def _printAttribute(obj: object):
  """
  打印对象的属性

  :param obj:
  :return:
  """
  print(f"[{obj.__class__.__name__} - 对象属性]".center(50, "-"))
  try:
    attrs = obj.__dict__
    if attrs:
      max_key_len = max(len(k) for k in attrs.keys())
      max_type_len = max(len(type(v).__name__) for v in attrs.values())
      for key, value in attrs.items():
        # 格式化输出，使属性名对齐
        print(f"  {key.ljust(max_key_len + 1)}: {type(value).__name__.ljust(max_type_len + 1)} : {value!r}")
    else:
      print("  该对象没有属性")
  except AttributeError:
    print("  无法获取对象属性")


def printClass(obj: object):
  _printClass(obj)
  print("-" * 50)


def printAttribute(obj: object):
  _printAttribute(obj)
  print("-" * 50)


def printObject(obj):
  """
  清晰打印对象的继承结构和属性信息

  参数:
      obj: 要检查的对象
  """
  # 打印对象类型标题
  print(obj.__class__.__name__.center(50, "="))
  print(f"对象信息: {obj!r}")
  _printClass(obj)
  _printAttribute(obj)
  print("=" * 50 + "\n")

