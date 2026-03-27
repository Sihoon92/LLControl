"""
CQL (Conservative Q-Learning) 학습 스크립트

d3rlpy 기반 Offline RL 학습 및 평가
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

import numpy as np
import d3rlpy

# 프로젝트 루트를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from apc_optimization.cql.config import (
    CQL_ALGORITHM_CONFIG,
    NETWORK_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    CQL_OUTPUT_DIR,
    CQL_MODEL_DIR,
    CQL_LOG_DIR,
    get_model_save_path,
    get_config_summary,
)
from apc_optimization.cql.data_processor import CQLDataProcessor


def setup_logging(output_dir: Path) -> logging.Logger:
    """로깅 설정"""
    log_file = output_dir / f"train_cql_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger('CQL_Training')
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"로그 파일: {log_file}")

    return logger


class CQLTrainer:
    """CQL (Conservative Q-Learning) 학습 관리자"""

    def __init__(
        self,
        data_file: str,
        output_dir: Path,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        """
        Args:
            data_file: MDP 데이터 파일 경로 (parquet/xlsx)
            output_dir: 출력 디렉토리
            device: 'cpu', 'cuda', 'mps'
            random_seed: 랜덤 시드
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.device = device
        self.random_seed = random_seed

        # 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        # 로거 설정
        self.logger = setup_logging(self.output_dir)

        # 랜덤 시드
        np.random.seed(random_seed)

        # 데이터 처리기
        self.data_processor = None
        self.data_dict = None

        # 모델
        self.cql_model = None

        # 학습 히스토리
        self.history = {
            'epoch_losses': [],
        }

    # ========================================================================
    # 데이터 준비
    # ========================================================================

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        self.logger.info("=" * 80)
        self.logger.info("데이터 로드 및 전처리")
        self.logger.info("=" * 80)

        self.data_processor = CQLDataProcessor(
            validation_split=TRAINING_CONFIG['validation_split'],
            random_seed=self.random_seed
        )

        self.data_dict = self.data_processor.process(self.data_file)

        self.logger.info(f"Observation 차원: {self.data_dict['obs_dim']}")
        self.logger.info(f"Action 차원: {self.data_dict['act_dim']}")
        self.logger.info(f"총 Transitions: {self.data_dict['n_transitions']}")
        self.logger.info(f"State 칼럼: {self.data_dict['state_columns'][:5]}...")
        self.logger.info(f"Action 칼럼: {self.data_dict['action_columns']}")

        # 스케일러 저장
        scaler_path = self.output_dir / "scaler.pkl"
        self.data_processor.save_scaler(str(scaler_path))
        self.logger.info(f"스케일러 저장: {scaler_path}")

    # ========================================================================
    # 모델 생성
    # ========================================================================

    def create_model(self):
        """CQL 모델 생성"""
        self.logger.info("=" * 80)
        self.logger.info("CQL 모델 생성")
        self.logger.info("=" * 80)

        self.cql_model = d3rlpy.algos.CQLConfig(
            actor_learning_rate=CQL_ALGORITHM_CONFIG['actor_learning_rate'],
            critic_learning_rate=CQL_ALGORITHM_CONFIG['critic_learning_rate'],
            alpha_learning_rate=CQL_ALGORITHM_CONFIG['alpha_learning_rate'],
            alpha_threshold=CQL_ALGORITHM_CONFIG['alpha_threshold'],
            conservative_weight=CQL_ALGORITHM_CONFIG['conservative_weight'],
            n_action_samples=CQL_ALGORITHM_CONFIG['n_action_samples'],
            batch_size=CQL_ALGORITHM_CONFIG['batch_size'],
            gamma=CQL_ALGORITHM_CONFIG['gamma'],
            tau=CQL_ALGORITHM_CONFIG['tau'],
        ).create(device=self.device)

        self.logger.info(f"CQL 모델 생성 완료 (device={self.device})")
        self.logger.info(f"  Actor LR: {CQL_ALGORITHM_CONFIG['actor_learning_rate']}")
        self.logger.info(f"  Critic LR: {CQL_ALGORITHM_CONFIG['critic_learning_rate']}")
        self.logger.info(f"  Conservative Weight: {CQL_ALGORITHM_CONFIG['conservative_weight']}")
        self.logger.info(f"  Batch Size: {CQL_ALGORITHM_CONFIG['batch_size']}")

    # ========================================================================
    # 학습
    # ========================================================================

    def train(self):
        """CQL 모델 학습"""
        self.logger.info("=" * 80)
        self.logger.info("CQL 학습 시작")
        self.logger.info("=" * 80)

        n_steps = TRAINING_CONFIG['n_steps']
        n_steps_per_epoch = TRAINING_CONFIG['n_steps_per_epoch']

        self.logger.info(f"Total Steps: {n_steps}")
        self.logger.info(f"Steps per Epoch: {n_steps_per_epoch}")
        self.logger.info(f"Total Epochs: {n_steps // n_steps_per_epoch}")

        train_dataset = self.data_dict['train_dataset']

        # d3rlpy 학습 실행
        self.cql_model.fit(
            train_dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            logger_adapter=d3rlpy.logging.FileAdapterFactory(
                root_dir=str(self.output_dir / "d3rlpy_logs")
            ),
            show_progress=True,
        )

        # 최고 모델 저장
        best_model_path = self.model_dir / "best_model.d3"
        self.cql_model.save(str(best_model_path))
        self.logger.info(f"모델 저장: {best_model_path}")

        # 모델 가중치만 별도 저장
        weights_path = self.model_dir / "best_model.pt"
        self.cql_model.save_model(str(weights_path))
        self.logger.info(f"모델 가중치 저장: {weights_path}")

        self.logger.info("=" * 80)
        self.logger.info("CQL 학습 완료")
        self.logger.info("=" * 80)

    # ========================================================================
    # 평가
    # ========================================================================

    def evaluate(self) -> dict:
        """학습된 모델 평가"""
        self.logger.info("=" * 80)
        self.logger.info("CQL 모델 평가")
        self.logger.info("=" * 80)

        val_dataset = self.data_dict['val_dataset']

        # Validation 데이터에서 예측
        val_observations = []
        val_actions = []
        val_rewards = []

        for episode in val_dataset.episodes:
            val_observations.append(episode.observations)
            val_actions.append(episode.actions)
            val_rewards.append(episode.rewards)

        val_obs = np.concatenate(val_observations, axis=0)
        val_act = np.concatenate(val_actions, axis=0)
        val_rew = np.concatenate(val_rewards, axis=0)

        # 모델 예측 액션
        predicted_actions = self.cql_model.predict(val_obs)

        # 메트릭 계산
        action_mse = float(np.mean((predicted_actions - val_act) ** 2))
        action_mae = float(np.mean(np.abs(predicted_actions - val_act)))

        # 예측 액션의 범위 통계
        act_min = float(predicted_actions.min())
        act_max = float(predicted_actions.max())
        act_mean = float(predicted_actions.mean())
        act_std = float(predicted_actions.std())

        metrics = {
            'action_mse': action_mse,
            'action_mae': action_mae,
            'predicted_action_min': act_min,
            'predicted_action_max': act_max,
            'predicted_action_mean': act_mean,
            'predicted_action_std': act_std,
            'n_val_samples': len(val_obs),
            'avg_reward': float(val_rew.mean()),
        }

        self.logger.info(f"Action MSE (vs behavioral): {action_mse:.6f}")
        self.logger.info(f"Action MAE (vs behavioral): {action_mae:.6f}")
        self.logger.info(f"Predicted Action 범위: [{act_min:.4f}, {act_max:.4f}]")
        self.logger.info(f"Predicted Action 평균: {act_mean:.4f} ± {act_std:.4f}")
        self.logger.info(f"Validation 평균 Reward: {val_rew.mean():.4f}")

        return metrics

    # ========================================================================
    # 결과 저장
    # ========================================================================

    def save_results(self, eval_metrics: dict):
        """결과 저장"""
        self.logger.info("=" * 80)
        self.logger.info("결과 저장")
        self.logger.info("=" * 80)

        # 1. 평가 메트릭 저장
        metrics_path = self.output_dir / "eval_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"평가 메트릭 저장: {metrics_path}")

        # 2. 설정 저장
        config_path = self.output_dir / "config.json"
        config = {
            'algorithm': CQL_ALGORITHM_CONFIG,
            'network': NETWORK_CONFIG,
            'training': TRAINING_CONFIG,
            'data': DATA_CONFIG,
            'device': self.device,
            'random_seed': self.random_seed,
            'obs_dim': self.data_dict['obs_dim'],
            'act_dim': self.data_dict['act_dim'],
            'state_columns': self.data_dict['state_columns'],
            'action_columns': self.data_dict['action_columns'],
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"설정 저장: {config_path}")

        # 3. 요약 리포트
        report_path = self.output_dir / "training_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CQL (Conservative Q-Learning) 학습 리포트\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"데이터 파일: {self.data_file}\n")
            f.write(f"출력 디렉토리: {self.output_dir}\n")
            f.write(f"디바이스: {self.device}\n")
            f.write(f"랜덤 시드: {self.random_seed}\n\n")

            f.write(f"데이터 정보:\n")
            f.write(f"  Observation 차원: {self.data_dict['obs_dim']}\n")
            f.write(f"  Action 차원: {self.data_dict['act_dim']}\n")
            f.write(f"  총 Transitions: {self.data_dict['n_transitions']}\n\n")

            f.write(f"알고리즘 설정:\n")
            f.write(f"  Actor LR: {CQL_ALGORITHM_CONFIG['actor_learning_rate']}\n")
            f.write(f"  Critic LR: {CQL_ALGORITHM_CONFIG['critic_learning_rate']}\n")
            f.write(f"  Conservative Weight: {CQL_ALGORITHM_CONFIG['conservative_weight']}\n")
            f.write(f"  Batch Size: {CQL_ALGORITHM_CONFIG['batch_size']}\n")
            f.write(f"  Total Steps: {TRAINING_CONFIG['n_steps']}\n\n")

            f.write(f"평가 결과:\n")
            for k, v in eval_metrics.items():
                f.write(f"  {k}: {v}\n")

            f.write("\n" + "=" * 80 + "\n")

        self.logger.info(f"학습 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='CQL (Conservative Q-Learning) 학습 및 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 학습
  python -m apc_optimization.cql.train

  # 데이터 파일 지정
  python -m apc_optimization.cql.train --data-file outputs/offline_rl_training_data.parquet

  # GPU 사용
  python -m apc_optimization.cql.train --device cuda
        """
    )

    parser.add_argument(
        '--data-file', type=str, default=None,
        help='MDP 데이터 파일 경로 (None이면 기본 경로 사용)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='출력 디렉토리 (None이면 기본 경로 사용)'
    )
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu',
        help='디바이스'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42,
        help='랜덤 시드'
    )

    args = parser.parse_args()

    # 데이터 파일 경로 결정
    if args.data_file is None:
        data_file = PROJECT_ROOT / 'outputs' / 'offline_rl_training_data.parquet'
    else:
        data_file = Path(args.data_file)

    # 출력 디렉토리 결정
    if args.output_dir is None:
        output_dir = CQL_OUTPUT_DIR / "training"
    else:
        output_dir = Path(args.output_dir)

    # 헤더 출력
    print("=" * 80)
    print("CQL (Conservative Q-Learning) 학습")
    print("=" * 80)
    print(f"데이터 파일: {data_file}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"디바이스: {args.device}")
    print("=" * 80)
    print()

    # 데이터 파일 존재 확인
    if not data_file.exists():
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다: {data_file}")
        print("먼저 전처리 파이프라인을 실행하여 offline_rl_training_data.parquet를 생성하세요.")
        return 1

    # Trainer 생성
    trainer = CQLTrainer(
        data_file=str(data_file),
        output_dir=output_dir,
        device=args.device,
        random_seed=args.random_seed
    )

    # Step 1: 데이터 로드
    trainer.load_and_prepare_data()

    # Step 2: 모델 생성
    trainer.create_model()

    # Step 3: 학습
    trainer.train()

    # Step 4: 평가
    eval_metrics = trainer.evaluate()

    # Step 5: 결과 저장
    trainer.save_results(eval_metrics)

    # 완료
    print("\n" + "=" * 80)
    print("모든 작업 완료!")
    print("=" * 80)
    print(f"\n데이터 파일: {data_file}")
    print(f"결과 저장 위치: {output_dir}")
    print("\n생성된 파일:")
    print("  - models/best_model.d3: CQL 모델 (전체)")
    print("  - models/best_model.pt: CQL 모델 가중치")
    print("  - scaler.pkl: 데이터 정규화 스케일러")
    print("  - eval_metrics.json: 평가 메트릭")
    print("  - config.json: 설정")
    print("  - training_report.txt: 종합 리포트")
    print()

    print("=" * 80)
    print("평가 결과:")
    print("=" * 80)
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
