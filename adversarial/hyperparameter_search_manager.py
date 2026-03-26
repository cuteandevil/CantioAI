"""
超参数搜索管理器：
负责管理和执行超参数搜索实验
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import os
from pathlib import Path
import copy
import datetime

# Try to import Optuna, but handle gracefully if not available
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Try to import Ray Tune, but handle gracefully if not available
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    tune = None

logger = logging.getLogger(__name__)

class HyperparameterSearchManager:
    """
    超参数搜索管理器：
    负责管理和执行超参数搜索实验
    """

    def __init__(self, base_config: Dict, objective_fn: Callable, experiment_dir: Optional[str] = None):
        """
        初始化超参数搜索管理器

        参数:
            base_config: 基础配置字典
            objective_fn: 目标函数，接受超参数配置并返回要优化的指标
            experiment_dir: 实验目录（如果不指定，则自动创建）
        """
        self.base_config = base_config
        self.objective_fn = objective_fn
        self.logger = logging.getLogger(__name__)

        # 设置实验目录
        if experiment_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = f"experiments/{self.base_config['experiment']['name']}_hp_search_{timestamp}"

        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 超参数搜索配置
        self.search_config = self.base_config.get("hyperparameter_search", {})
        self.enabled = self.search_config.get("enabled", False)
        self.method = self.search_config.get("method", "optuna")
        self.study_name = self.search_config.get("study_name", "cantioai_hyperparameter_search")
        self.storage = self.search_config.get("storage", "sqlite:///optuna_study.db")
        self.n_trials = self.search_config.get("n_trials", 100)
        self.timeout = self.search_config.get("timeout", 3600)
        self.direction = self.search_config.get("direction", "minimize")
        self.sampler_name = self.search_config.get("sampler", "TPESampler")
        self.pruner_name = self.search_config.get("pruner", "MedianPruner")

        self.logger.info(f"超参数搜索管理器初始化完成，方法: {self.method}, 启用: {self.enabled}")

    def _setup_logging(self):
        """设置日志系统"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # 配置文件日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "hyperparameter_search.log"),
                logging.StreamHandler()
            ]
        )

        # 保存基础配置
        config_save_path = self.experiment_dir / "base_config.yaml"
        try:
            import yaml
            with open(config_save_path, 'w') as f:
                yaml.dump(self.base_config, f, default_flow_style=False)
        except ImportError:
            self.logger.warning("YAML not available, skipping config saving")

    def _get_sampler(self):
        """获取Optuna采样器"""
        if not OPTUNA_AVAILABLE:
            return None

        sampler_map = {
            "TPESampler": TPESampler,
            "RandomSampler": RandomSampler,
        }
        sampler_class = sampler_map.get(self.sampler_name, TPESampler)
        return sampler_class()

    def _get_pruner(self):
        """获取Optuna剪枝器"""
        if not OPTUNA_AVAILABLE:
            return None

        pruner_map = {
            "MedianPruner": MedianPruner,
            "SuccessiveHalvingPruner": SuccessiveHalvingPruner,
        }
        pruner_class = pruner_map.get(self.pruner_name, MedianPruner)
        return pruner_class()

    def define_search_space(self, trial):
        """
        定义超参数搜索空间
        这个方法应该被子类重写以定义具体的搜索空间

        参数:
            trial: Optuna试验对象

        返回:
            超参数配置字典
        """
        # 默认实现：返回空配置，子类应该重写此方法
        self.logger.warning("define_search_space method not overridden, returning empty config")
        return {}

    def objective_wrapper(self, trial):
        """
        Optuna目标函数包装器

        参数:
            trial: Optuna试验对象

        返回:
            要优化的指标值
        """
        try:
            # 定义搜索空间并获取超参数
            search_space_params = self.define_search_space(trial)

            # 创建带有超参数的配置
            config = copy.deepcopy(self.base_config)
            self._update_config_with_params(config, search_space_params)

            # 运行目标函数
            metric_value = self.objective_fn(config)

            # 记录试验信息
            self.logger.info(f"Trial {trial.number} finished with value: {metric_value} and params: {search_space_params}")

            return metric_value
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {e}")
            # 返回一个很坏的值以便Optuna知道这次试验失败了
            if self.direction == "minimize":
                return float('inf')
            else:
                return float('-inf')

    def _update_config_with_params(self, config: Dict, params: Dict):
        """
        用超参数更新配置

        参数:
            config: 要更新的配置字典
            params: 超参数字典
        """
        # 简单实现：直接更新配置
        # 更复杂的实现可以支持嵌套键路径等
        for key, value in params.items():
            if '.' in key:
                # 处理嵌套键，如 "training.learning_rate"
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = value
            else:
                config[key] = value

    def run_search(self):
        """
        运行超参数搜索

        返回:
            最佳试验结果
        """
        if not self.enabled:
            self.logger.info("超参数搜索未启用，跳过搜索")
            return None

        self.logger.info(f"开始超参数搜索，方法: {self.method}, 试验次数: {self.n_trials}")

        if self.method == "optuna":
            return self._run_optuna_search()
        elif self.method == "ray":
            return self._run_ray_search()
        elif self.method == "grid":
            return self._run_grid_search()
        else:
            self.logger.error(f"不支持的超参数搜索方法: {self.method}")
            return None

    def _run_optuna_search(self):
        """运行Optuna超参数搜索"""
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna未安装，无法运行Optuna搜索。请安装: pip install optuna")
            return None

        try:
            # 创建study
            sampler = self._get_sampler()
            pruner = self._get_pruner()

            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction=self.direction,
                sampler=sampler,
                pruner=pruner
            )

            # 优化目标函数
            study.optimize(
                self.objective_wrapper,
                n_trials=self.n_trials,
                timeout=self.timeout
            )

            # 记录结果
            self.logger.info(f"超参数搜索完成，最佳值: {study.best_value}")
            self.logger.info(f"最佳参数: {study.best_params}")

            # 保存研究结果
            self._save_study_results(study)

            return study.best_trial
        except Exception as e:
            self.logger.error(f"Optuna搜索失败: {e}")
            return None

    def _run_ray_search(self):
        """运行Ray Tune超参数搜索"""
        if not RAY_AVAILABLE:
            self.logger.error("Ray Tune未安装，无法运行Ray Tune搜索。请安装: pip install ray[tune]")
            return None

        self.logger.warning("Ray Tune搜索尚未完全实现")
        return None

    def _run_grid_search(self):
        """运行网格搜索"""
        self.logger.warning("网格搜索尚未完全实现")
        return None

    def _save_study_results(self, study):
        """保存研究结果"""
        try:
            # 保存最佳试验信息
            best_trial_info = {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "datetime_start": study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
                "datetime_complete": study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None,
            }

            import json
            with open(self.experiment_dir / "best_trial.json", 'w') as f:
                json.dump(best_trial_info, f, indent=2)

            # 保存所有试验信息
            trials_info = []
            for trial in study.trials:
                trial_info = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    "state": str(trial.state),
                }
                trials_info.append(trial_info)

            with open(self.experiment_dir / "all_trials.json", 'w') as f:
                json.dump(trials_info, f, indent=2)

            self.logger.info(f"研究结果已保存到 {self.experiment_dir}")
        except Exception as e:
            self.logger.error(f"保存研究结果失败: {e}")

    def get_best_config(self, study=None):
        """
        获取最佳配置

        参数:
            study: Optuna研究对象（如果为None，则尝试加载现有研究）

        返回:
            最佳配置字典
        """
        if study is None and self.method == "optuna" and OPTUNA_AVAILABLE:
            try:
                study = optuna.load_study(study_name=self.study_name, storage=self.storage)
            except Exception as e:
                self.logger.error(f"加载研究失败: {e}")
                return None

        if study is not None and hasattr(study, 'best_trial'):
            best_params = study.best_params
            config = copy.deepcopy(self.base_config)
            self._update_config_with_params(config, best_params)
            return config
        else:
            self.logger.warning("无法获取最佳配置")
            return None

    def close(self):
        """关闭资源"""
        self.logger.info("超参数搜索管理器已关闭")