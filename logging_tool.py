import os
import argparse
import yaml
import logging
from datetime import datetime
from typing import Dict, Any
import sys
class DualOutput:#获取控制台的输出转化为日志文件
    """同时向文件和控制台输出"""

    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, obj):
        # 写入文件
        self.file.write(obj)
        # 写入标准输出
        self.stdout.write(obj)
        # 实时刷新缓冲区
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()
def get_nested_value(config, path):
    current = config
    for key in path.split('.'):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lower = value.lower()
    if lower in ('yes', 'true', 't', 'y', ):
        return True
    elif lower in ('no', 'false', 'f', 'n', ):
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean: {value}')
def setup_logging(config: Dict[str, Any]) -> str:
    """配置日志系统"""
    log_dir = config["logging"]["log_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experimental_args=''
    for item in config['Experimental_args']:
        experimental_args+=str(item)
        keys = item.split(".")
        data_item = config[keys[0]]
        for i in range(len(keys) - 1):
            data_item = data_item[keys[i + 1]]
        experimental_args+='='+str(data_item)
    print("测试节点 experimental_args",experimental_args)
    log_dir = os.path.join(log_dir, config["Experimental_purpose"],experimental_args)
    print("测试节点 log_dir",log_dir)
    os.makedirs(log_dir, exist_ok=True)


    log_filename = f"run_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    print("测试节点 log_path",log_path)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')  # 日志文件编码
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(config["logging"]["level"].upper())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_dir


# def load_config(config_path: str) -> Dict[str, Any]:
#     """加载YAML配置文件（修正编码问题）"""
#     try:
#         # 显式指定编码为UTF-8
#         with open(config_path, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#         logging.info(f"✅ 成功加载配置文件: {config_path}")
#         return config
#     except UnicodeDecodeError as e:
#         error_msg = f"编码错误: {e.reason}\n建议操作：\n1. 使用Notepad++打开文件\n2. 点击菜单Encoding → Convert to UTF-8-BOM\n3. 保存文件"
#         logging.error(f"❌ {error_msg}")
#         raise
#     except FileNotFoundError:
#         logging.error(f"❌ 配置文件不存在: {config_path}")
#         raise
#     except yaml.YAMLError as e:
#         logging.error(f"❌ YAML解析错误: {str(e)}")
#         raise
def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件（支持多编码）"""
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'utf-16']  # 按优先级尝试的编码列表

    for encoding in encodings_to_try:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                config = yaml.safe_load(f)
            logging.info(f"成功加载配置文件 [{encoding}]: {config_path}")
            return config
        except UnicodeDecodeError:
            continue  # 尝试下一种编码
        except yaml.YAMLError as e:
            logging.error(f"YAML解析错误: {str(e)}")
            raise

    # 所有编码尝试失败
    error_msg = (
        f"无法解码文件 {config_path}\n"
        "可能原因：\n"
        "1. 文件包含非文本内容（如二进制数据）\n"
        "2. 使用了非标准编码\n"
        "解决方案：\n"
        "1. 用 Notepad++ 打开文件 → Encoding → Convert to UTF-8\n"
        "2. 检查文件是否损坏"
    )
    logging.error(error_msg)
    raise UnicodeDecodeError("无法用任何候选编码解码文件")

def merge_configs(base_cfg: Dict, cli_cfg: Dict) -> Dict:
    """深度合并配置字典"""

    def recursive_merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = recursive_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    return recursive_merge(base_cfg.copy(), cli_cfg)


def parse_args(config=None) -> Dict[str, Any]:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="配置管理系统示例")

    # 必需参数
    if config is None:
        parser.add_argument("--config", type=str, required=True,
                        help="基础配置文件路径")
    else:
        parser.add_argument("--config",type=str,default=config,help="基础配置文件路劲")

    # 可覆盖参数示例
    parser.add_argument("--data.batch_size", type=int,
                        help="训练的batch_size")
    parser.add_argument("--model.lr", type=float,
                        help="覆盖学习率")
    # parser.add_argument("--training.epochs", type=int,
    #                     help="覆盖训练轮数")
    # parser.add_argument("--logging.level", type=str,
    #                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    #                     help="覆盖日志级别")
    print("测试节点 parser",parser.parse_args())
    # 解析为嵌套字典
    args = parser.parse_args()
    cli_config = {}
    for arg_key, arg_value in vars(args).items():
        if arg_value is None:
            continue
        keys = arg_key.split('.')
        current = cli_config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = arg_value
    print("测试节点 cli_config",cli_config)
    return cli_config


def save_final_config(config: Dict, log_dir: str,time:str) -> None:
    """保存最终配置到日志目录"""
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if time is None:
        config_path = os.path.join(log_dir, f"config.yaml")
    else:
        config_path=os.path.join(log_dir, f"config{time}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(
            config,
            f,
            allow_unicode=True,  # 🎯 禁止转义中文
            default_flow_style=False,
            sort_keys=False  # 保持键顺序
        )

    logging.info(f" 配置已保存至: {config_path}")


def main():
    # 初始化基础日志（用于参数解析阶段）
    logging.basicConfig(level=logging.INFO)

    # 解析参数
    cli_config = parse_args("./configs/base.yaml")
    base_config = load_config(cli_config["config"])
    #print("测试节点",base_config)
    # 合并配置
    final_config = merge_configs(base_config, cli_config)

    # 初始化正式日志系统
    log_dir= setup_logging(final_config)
    logging.info("=" * 50)
    # logging.info(" 程序启动 - 最终配置参数:")
    # logging.info(yaml.dump(final_config, default_flow_style=False))

    # 保存配置副本
    save_final_config(final_config, log_dir)

    # 以下是业务逻辑示例
    # logging.info(f" 输入目录: {final_config['data']['input_dir']}")
    # logging.info(f"️ 学习率: {final_config['model']['lr']}")
    # logging.info(f" 训练轮数: {final_config['training']['epochs']}")


if __name__ == "__main__":
    main()