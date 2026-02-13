"""
MBRL (PETS) 모델 학습 스크립트

Per-Zone Probabilistic Ensemble 학습 및 평가
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

import numpy as np
import torch

# 프로젝트 루트를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apc_optimization.mbrl import (
    PerZoneProbabilisticEnsemble,
    PETSDataProcessor,
    DYNAMICS_MODEL_CONFIG,
    ENSEMBLE_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    MBRL_OUTPUT_DIR,
    MBRL_MODEL_DIR,
    MBRL_LOG_DIR,
    get_model_save_path,
    get_config_summary,
)


def setup_logging(output_dir: Path, mode: str) -> logging.Logger:
    """로깅 설정"""
    log_file = output_dir / f"train_mbrl_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 로거 생성
    logger = logging.getLogger('MBRL_Training')
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

    # 핸들러 추가
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"로그 파일: {log_file}")

    return logger


class MBRLTrainer:
    """MBRL (PETS) 학습 관리자"""

    def __init__(
        self,
        data_file: str,
        output_dir: Path,
        mode: str = 'training',
        device: str = 'cpu',
        random_seed: int = 42
    ):
        """
        Args:
            data_file: 학습 데이터 파일 경로
            output_dir: 출력 디렉토리
            mode: 'training' 또는 'test'
            device: 'cpu', 'cuda', 'mps'
            random_seed: 랜덤 시드
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.device = device
        self.random_seed = random_seed

        # 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        # 로거 설정
        self.logger = setup_logging(self.output_dir, mode)

        # 랜덤 시드 고정
        self._set_random_seed(random_seed)

        # 데이터 처리기
        self.data_processor = None
        self.data_dict = None

        # 모델
        self.model = None

        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'train_mse': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': [],
            'val_uncertainty': [],
        }

        # 최고 성능
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _set_random_seed(self, seed: int):
        """랜덤 시드 고정"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"랜덤 시드 설정: {seed}")

    # ========================================================================
    # 데이터 준비
    # ========================================================================

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        self.logger.info("="*80)
        self.logger.info("데이터 로드 및 전처리")
        self.logger.info("="*80)

        # 데이터 처리기 생성
        self.data_processor = PETSDataProcessor(
            normalize=DATA_CONFIG['normalize_inputs'],
            validation_split=TRAINING_CONFIG['validation_split'],
            test_split=TRAINING_CONFIG['test_split'],
            random_seed=self.random_seed
        )

        # 데이터 처리
        self.data_dict = self.data_processor.process(
            data_file=self.data_file,
            mode=self.mode
        )

        # 데이터 정보 출력
        X_train, y_train = self.data_dict['train']
        X_val, y_val = self.data_dict['val']
        X_test, y_test = self.data_dict['test']

        self.logger.info(f"Train: {len(X_train)} 샘플")
        self.logger.info(f"Val:   {len(X_val)} 샘플")
        self.logger.info(f"Test:  {len(X_test)} 샘플")
        self.logger.info(f"입력 차원: {X_train.shape[1]}")
        self.logger.info(f"출력 차원: {y_train.shape[1]}")

        # Scaler 저장
        scaler_path = self.output_dir / "scaler.pkl"
        self.data_processor.save_scaler(str(scaler_path))
        self.logger.info(f"Scaler 저장: {scaler_path}")

    # ========================================================================
    # 모델 학습
    # ========================================================================

    def create_model(self):
        """모델 생성"""
        self.logger.info("="*80)
        self.logger.info("모델 생성")
        self.logger.info("="*80)

        self.model = PerZoneProbabilisticEnsemble(
            n_ensembles=ENSEMBLE_CONFIG['n_ensembles'],
            input_dim=DYNAMICS_MODEL_CONFIG['input_dim'],
            output_dim=DYNAMICS_MODEL_CONFIG['output_dim'],
            hidden_dims=DYNAMICS_MODEL_CONFIG['hidden_dims'],
            device=self.device,
            activation=DYNAMICS_MODEL_CONFIG['activation'],
            use_layer_norm=DYNAMICS_MODEL_CONFIG['use_layer_norm'],
            dropout=DYNAMICS_MODEL_CONFIG['dropout'],
            log_var_min=TRAINING_CONFIG['log_var_min'],
            log_var_max=TRAINING_CONFIG['log_var_max'],
        )

        # Optimizer 초기화
        self.model.init_optimizers(
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        # 모델 정보
        info = self.model.get_model_info()
        self.logger.info(f"앙상블 개수: {info['n_ensembles']}")
        self.logger.info(f"총 파라미터: {info['total_parameters']:,}")
        self.logger.info(f"모델당 파라미터: {info['parameters_per_model']:,}")

    def train(self):
        """모델 학습"""
        self.logger.info("="*80)
        self.logger.info("모델 학습 시작")
        self.logger.info("="*80)

        X_train, y_train = self.data_dict['train']
        X_val, y_val = self.data_dict['val']

        batch_size = TRAINING_CONFIG['batch_size']
        epochs = TRAINING_CONFIG['epochs']
        early_stopping_patience = TRAINING_CONFIG['early_stopping_patience']

        n_batches = len(X_train) // batch_size
        patience_counter = 0

        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Batches per Epoch: {n_batches}")

        for epoch in range(epochs):
            # ========== Training ==========
            epoch_train_loss = 0.0
            epoch_train_mse = 0.0

            # 배치 샘플링
            indices = np.random.permutation(len(X_train))

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # 학습 step
                metrics = self.model.train_on_batch(X_batch, y_batch)

                epoch_train_loss += metrics['loss']
                epoch_train_mse += metrics['mse']

            # 평균
            epoch_train_loss /= n_batches
            epoch_train_mse /= n_batches

            # ========== Validation ==========
            val_metrics = self.model.evaluate_on_batch(X_val, y_val)

            # 히스토리 저장
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_mse'].append(epoch_train_mse)
            self.history['val_loss'].append(val_metrics['mse'])  # 평가는 MSE로
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_uncertainty'].append(val_metrics['mean_uncertainty'])

            # 로깅
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"Train Loss={epoch_train_loss:.6f}, "
                    f"Val MSE={val_metrics['mse']:.6f}, "
                    f"Val R²={val_metrics['r2']:.4f}"
                )

            # 최고 성능 체크
            if val_metrics['mse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['mse']
                self.best_epoch = epoch
                patience_counter = 0

                # 최고 모델 저장
                best_model_path = self.model_dir / "best_model.pt"
                self.model.save(str(best_model_path))
                self.logger.info(f"  → 최고 성능 갱신! (Epoch {epoch+1}, Val MSE={val_metrics['mse']:.6f})")

            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early Stopping at Epoch {epoch+1} (Patience: {early_stopping_patience})")
                break

            # 체크포인트 저장 (주기적)
            if (epoch + 1) % TRAINING_CONFIG['checkpoint_interval'] == 0:
                checkpoint_path = self.model_dir / f"checkpoint_epoch{epoch+1}.pt"
                self.model.save(str(checkpoint_path))

        self.logger.info("="*80)
        self.logger.info(f"학습 완료! 최고 성능: Epoch {self.best_epoch+1}, Val MSE={self.best_val_loss:.6f}")
        self.logger.info("="*80)

    # ========================================================================
    # 평가
    # ========================================================================

    def evaluate(self):
        """최종 평가"""
        self.logger.info("="*80)
        self.logger.info("최종 평가 (Test 데이터)")
        self.logger.info("="*80)

        # 최고 모델 로드
        best_model_path = self.model_dir / "best_model.pt"
        self.model.load(str(best_model_path))

        X_test, y_test = self.data_dict['test']

        # 평가
        test_metrics = self.model.evaluate_on_batch(X_test, y_test)

        self.logger.info(f"Test MSE:  {test_metrics['mse']:.6f}")
        self.logger.info(f"Test MAE:  {test_metrics['mae']:.6f}")
        self.logger.info(f"Test RMSE: {test_metrics['rmse']:.6f}")
        self.logger.info(f"Test R²:   {test_metrics['r2']:.4f}")
        self.logger.info(f"Mean Uncertainty: {test_metrics['mean_uncertainty']:.6f}")

        return test_metrics

    # ========================================================================
    # 결과 저장
    # ========================================================================

    def save_results(self, test_metrics: dict):
        """결과 저장"""
        self.logger.info("="*80)
        self.logger.info("결과 저장")
        self.logger.info("="*80)

        # 1. 학습 히스토리 저장
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"학습 히스토리 저장: {history_path}")

        # 2. 최종 메트릭 저장
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        self.logger.info(f"테스트 메트릭 저장: {metrics_path}")

        # 3. 설정 저장
        config_path = self.output_dir / "config.json"
        config = {
            'dynamics_model': DYNAMICS_MODEL_CONFIG,
            'ensemble': ENSEMBLE_CONFIG,
            'training': TRAINING_CONFIG,
            'data': DATA_CONFIG,
            'mode': self.mode,
            'device': self.device,
            'random_seed': self.random_seed,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"설정 저장: {config_path}")

        # 4. 요약 리포트
        report_path = self.output_dir / "training_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MBRL (PETS) 학습 리포트\n")
            f.write("="*80 + "\n\n")

            f.write(f"모드: {self.mode}\n")
            f.write(f"데이터 파일: {self.data_file}\n")
            f.write(f"출력 디렉토리: {self.output_dir}\n")
            f.write(f"디바이스: {self.device}\n")
            f.write(f"랜덤 시드: {self.random_seed}\n\n")

            f.write("모델 설정:\n")
            f.write(f"  앙상블 개수: {ENSEMBLE_CONFIG['n_ensembles']}\n")
            f.write(f"  Hidden Dims: {DYNAMICS_MODEL_CONFIG['hidden_dims']}\n")
            f.write(f"  Activation: {DYNAMICS_MODEL_CONFIG['activation']}\n\n")

            f.write("학습 설정:\n")
            f.write(f"  Epochs: {TRAINING_CONFIG['epochs']}\n")
            f.write(f"  Batch Size: {TRAINING_CONFIG['batch_size']}\n")
            f.write(f"  Learning Rate: {TRAINING_CONFIG['learning_rate']}\n")
            f.write(f"  Weight Decay: {TRAINING_CONFIG['weight_decay']}\n\n")

            f.write("학습 결과:\n")
            f.write(f"  최고 Epoch: {self.best_epoch + 1}\n")
            f.write(f"  최고 Val MSE: {self.best_val_loss:.6f}\n\n")

            f.write("최종 평가 (Test):\n")
            f.write(f"  MSE:  {test_metrics['mse']:.6f}\n")
            f.write(f"  MAE:  {test_metrics['mae']:.6f}\n")
            f.write(f"  RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"  R²:   {test_metrics['r2']:.4f}\n")
            f.write(f"  Mean Uncertainty: {test_metrics['mean_uncertainty']:.6f}\n\n")

            f.write("="*80 + "\n")

        self.logger.info(f"학습 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""

    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='MBRL (PETS) 모델 학습 및 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Training 모드 (기본)
  python -m apc_optimization.mbrl.train
  python -m apc_optimization.mbrl.train --mode training

  # Test 모드
  python -m apc_optimization.mbrl.train --mode test

  # GPU 사용
  python -m apc_optimization.mbrl.train --device cuda

  # 데이터 파일 직접 지정
  python -m apc_optimization.mbrl.train --data-file outputs/custom_data.xlsx
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['training', 'test'],
        default='training',
        help='학습 모드 (training: 학습 데이터, test: 테스트 데이터)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='데이터 파일 경로 (None이면 mode 기반 자동 생성)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='출력 디렉토리 (None이면 기본 경로 사용)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cpu',
        help='디바이스 (cpu, cuda, mps)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )

    args = parser.parse_args()

    # 데이터 파일 경로 결정
    if args.data_file is None:
        data_file = PROJECT_ROOT / 'outputs' / f'model_{args.mode}_data.xlsx'
    else:
        data_file = Path(args.data_file)

    # 출력 디렉토리 결정
    if args.output_dir is None:
        output_dir = MBRL_OUTPUT_DIR / args.mode
    else:
        output_dir = Path(args.output_dir)

    # 헤더 출력
    print("="*80)
    print(f"MBRL (PETS) 모델 학습 ({args.mode.upper()} 모드)")
    print("="*80)
    print(f"데이터 파일: {data_file}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"디바이스: {args.device}")
    print("="*80)
    print()

    # 데이터 파일 존재 확인
    if not data_file.exists():
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다: {data_file}")
        print(f"먼저 전처리 파이프라인을 실행하여 model_{args.mode}_data.xlsx를 생성하세요.")
        print()
        print(f"전처리 실행 예시:")
        print(f"  python main.py --mode {args.mode} --apc data/raw/apc_data.xlsx --densitometer data/raw/densitometer_data.csv")
        return 1

    # PyTorch 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'

    # Trainer 생성
    trainer = MBRLTrainer(
        data_file=str(data_file),
        output_dir=output_dir,
        mode=args.mode,
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
    test_metrics = trainer.evaluate()

    # Step 5: 결과 저장
    trainer.save_results(test_metrics)

    # 완료
    print("\n" + "="*80)
    print("모든 작업 완료!")
    print("="*80)
    print(f"\n학습 모드: {args.mode.upper()}")
    print(f"데이터 파일: {data_file}")
    print(f"결과 저장 위치: {output_dir}")
    print("\n생성된 파일:")
    print("  - best_model.pt: 최고 성능 모델")
    print("  - training_history.json: 학습 히스토리")
    print("  - test_metrics.json: 테스트 메트릭")
    print("  - config.json: 설정")
    print("  - training_report.txt: 종합 리포트")
    print("  - scaler.pkl: 데이터 정규화 Scaler")
    print()

    print("="*80)
    print("최종 성능:")
    print("="*80)
    print(f"MSE:  {test_metrics['mse']:.6f}")
    print(f"MAE:  {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"R²:   {test_metrics['r2']:.4f}")
    print(f"Mean Uncertainty: {test_metrics['mean_uncertainty']:.6f}")
    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
