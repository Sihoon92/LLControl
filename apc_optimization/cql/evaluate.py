"""
CQL (Conservative Q-Learning) 평가 스크립트

학습된 CQL 모델을 로드하여 validation 데이터로 평가

사용법:
  # 기본 평가 (학습 시 사용한 MDP 데이터에서 validation split)
  python -m apc_optimization.cql.evaluate

  # 모델/데이터 경로 지정
  python -m apc_optimization.cql.evaluate \
      --model-path outputs/cql/training/models/best_model.d3 \
      --scaler-path outputs/cql/training/scaler.pkl \
      --data-file outputs/offline_rl_training_data.parquet

  # 결과를 파일로 저장
  python -m apc_optimization.cql.evaluate --output-dir outputs/cql/eval_results
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apc_optimization.cql.config import (
    CQL_OUTPUT_DIR,
    TRAINING_CONFIG,
    ACTION_CONSTRAINT_CONFIG,
)
from apc_optimization.cql.data_processor import CQLDataProcessor
from apc_optimization.cql.policy import CQLPolicy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logging(output_dir: Path = None) -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger('CQL_Evaluate')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        log_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def evaluate(
    model_path: str,
    scaler_path: str,
    data_file: str,
    output_dir: Path = None,
    random_seed: int = 42,
):
    """
    학습된 CQL 모델을 validation 데이터로 평가

    Args:
        model_path: 학습된 모델 경로 (.d3)
        scaler_path: 스케일러 경로 (.pkl)
        data_file: MDP 데이터 파일 경로
        output_dir: 결과 저장 디렉토리 (None이면 콘솔 출력만)
        random_seed: 랜덤 시드 (학습 시와 동일해야 동일한 val split)
    """
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("CQL 모델 평가 시작")
    logger.info("=" * 80)
    logger.info(f"모델: {model_path}")
    logger.info(f"스케일러: {scaler_path}")
    logger.info(f"데이터: {data_file}")
    logger.info(f"랜덤 시드: {random_seed}")

    # ================================================================
    # 1. 데이터 로드 및 validation split 재현
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 1: 데이터 로드 및 validation split")
    logger.info("-" * 80)

    data_processor = CQLDataProcessor(
        validation_split=TRAINING_CONFIG['validation_split'],
        random_seed=random_seed,
    )
    data_dict = data_processor.process(data_file)

    val_dataset = data_dict['val_dataset']
    state_columns = data_dict['state_columns']
    action_columns = data_dict['action_columns']

    # validation 데이터 추출
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

    logger.info(f"Validation 샘플 수: {len(val_obs)}")
    logger.info(f"Observation 차원: {val_obs.shape[1]}")
    logger.info(f"Action 차원: {val_act.shape[1]}")

    # ================================================================
    # 2. 모델 로드 및 예측
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 2: 모델 로드 및 예측")
    logger.info("-" * 80)

    import d3rlpy
    model = d3rlpy.load_learnable(model_path)
    logger.info("모델 로드 완료")

    # 정규화된 상태에서 바로 예측 (val_obs는 이미 정규화됨)
    predicted_actions_norm = model.predict(val_obs)

    # ================================================================
    # 3. 정규화 공간에서의 메트릭 (모델이 직접 출력하는 공간)
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 3: 정규화 공간 메트릭")
    logger.info("-" * 80)

    norm_mse = float(np.mean((predicted_actions_norm - val_act) ** 2))
    norm_mae = float(np.mean(np.abs(predicted_actions_norm - val_act)))

    logger.info(f"[정규화 공간] MSE: {norm_mse:.6f}")
    logger.info(f"[정규화 공간] MAE: {norm_mae:.6f}")

    # 칼럼별 메트릭 (정규화 공간)
    norm_per_action = {}
    for i, col in enumerate(action_columns):
        col_mse = float(np.mean((predicted_actions_norm[:, i] - val_act[:, i]) ** 2))
        col_mae = float(np.mean(np.abs(predicted_actions_norm[:, i] - val_act[:, i])))
        norm_per_action[col] = {'mse': col_mse, 'mae': col_mae}
        logger.info(f"  {col}: MSE={col_mse:.6f}, MAE={col_mae:.6f}")

    # ================================================================
    # 4. 원본 스케일 메트릭 (역정규화 후)
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 4: 원본 스케일 메트릭 (역정규화)")
    logger.info("-" * 80)

    pred_actions_orig = data_processor.inverse_transform_action(predicted_actions_norm)
    val_act_orig = data_processor.inverse_transform_action(val_act)

    orig_mse = float(np.mean((pred_actions_orig - val_act_orig) ** 2))
    orig_mae = float(np.mean(np.abs(pred_actions_orig - val_act_orig)))

    logger.info(f"[원본 스케일] MSE: {orig_mse:.6f}")
    logger.info(f"[원본 스케일] MAE: {orig_mae:.6f}")

    # 칼럼별 메트릭 (원본 스케일)
    orig_per_action = {}
    for i, col in enumerate(action_columns):
        col_mse = float(np.mean((pred_actions_orig[:, i] - val_act_orig[:, i]) ** 2))
        col_mae = float(np.mean(np.abs(pred_actions_orig[:, i] - val_act_orig[:, i])))
        col_pred_mean = float(pred_actions_orig[:, i].mean())
        col_pred_std = float(pred_actions_orig[:, i].std())
        col_actual_mean = float(val_act_orig[:, i].mean())
        col_actual_std = float(val_act_orig[:, i].std())
        orig_per_action[col] = {
            'mse': col_mse,
            'mae': col_mae,
            'pred_mean': col_pred_mean,
            'pred_std': col_pred_std,
            'actual_mean': col_actual_mean,
            'actual_std': col_actual_std,
        }
        logger.info(
            f"  {col}: MSE={col_mse:.6f}, MAE={col_mae:.6f} | "
            f"pred={col_pred_mean:.4f}±{col_pred_std:.4f}, "
            f"actual={col_actual_mean:.4f}±{col_actual_std:.4f}"
        )

    # ================================================================
    # 5. 제약 조건 적용 후 메트릭
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 5: 제약 조건 적용 후 메트릭")
    logger.info("-" * 80)

    policy = CQLPolicy(
        model_path=model_path,
        scaler_path=scaler_path,
    )

    # 원본 스케일 observation 복원 후 policy로 예측
    val_obs_orig = data_processor.inverse_transform_observation(val_obs)
    constrained_actions = policy.predict_batch(val_obs_orig)

    constrained_mse = float(np.mean((constrained_actions - val_act_orig) ** 2))
    constrained_mae = float(np.mean(np.abs(constrained_actions - val_act_orig)))

    # 제약 위반율
    n_violated = 0
    for i in range(len(val_obs_orig)):
        raw = pred_actions_orig[i]
        con = constrained_actions[i]
        if not np.allclose(raw, con, atol=1e-6):
            n_violated += 1
    violation_rate = n_violated / len(val_obs_orig) * 100

    logger.info(f"[제약 적용] MSE: {constrained_mse:.6f}")
    logger.info(f"[제약 적용] MAE: {constrained_mae:.6f}")
    logger.info(f"제약 위반 → 보정된 샘플: {n_violated}/{len(val_obs_orig)} ({violation_rate:.1f}%)")

    # ================================================================
    # 6. 액션 분포 통계
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 6: 액션 분포 요약")
    logger.info("-" * 80)

    logger.info(f"{'':>20s} | {'예측 (원본)':>20s} | {'실제 (원본)':>20s} | {'예측 (제약후)':>20s}")
    logger.info(f"{'칼럼':>20s} | {'min ~ max':>20s} | {'min ~ max':>20s} | {'min ~ max':>20s}")
    logger.info("-" * 90)
    for i, col in enumerate(action_columns):
        p_min, p_max = pred_actions_orig[:, i].min(), pred_actions_orig[:, i].max()
        a_min, a_max = val_act_orig[:, i].min(), val_act_orig[:, i].max()
        c_min, c_max = constrained_actions[:, i].min(), constrained_actions[:, i].max()
        logger.info(
            f"{col:>20s} | {p_min:+8.4f} ~ {p_max:+8.4f} | "
            f"{a_min:+8.4f} ~ {a_max:+8.4f} | {c_min:+8.4f} ~ {c_max:+8.4f}"
        )

    # ================================================================
    # 7. 보상 통계
    # ================================================================
    logger.info("-" * 80)
    logger.info("Step 7: Validation 보상 통계")
    logger.info("-" * 80)

    logger.info(f"평균 reward: {val_rew.mean():.4f}")
    logger.info(f"std reward:  {val_rew.std():.4f}")
    logger.info(f"min reward:  {val_rew.min():.4f}")
    logger.info(f"max reward:  {val_rew.max():.4f}")

    # ================================================================
    # 결과 종합
    # ================================================================
    results = {
        'n_val_samples': len(val_obs),
        'obs_dim': int(val_obs.shape[1]),
        'act_dim': int(val_act.shape[1]),
        'normalized_space': {
            'mse': norm_mse,
            'mae': norm_mae,
            'per_action': norm_per_action,
        },
        'original_scale': {
            'mse': orig_mse,
            'mae': orig_mae,
            'per_action': orig_per_action,
        },
        'constrained': {
            'mse': constrained_mse,
            'mae': constrained_mae,
            'violation_rate_pct': violation_rate,
            'n_violated': n_violated,
        },
        'reward_stats': {
            'mean': float(val_rew.mean()),
            'std': float(val_rew.std()),
            'min': float(val_rew.min()),
            'max': float(val_rew.max()),
        },
        'model_path': str(model_path),
        'data_file': str(data_file),
        'random_seed': random_seed,
    }

    # ================================================================
    # 결과 저장
    # ================================================================
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 메트릭 JSON
        metrics_path = output_dir / "eval_results.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logger.info(f"평가 결과 저장: {metrics_path}")

        # 예측 vs 실제 상세 CSV
        detail_data = {}
        for i, col in enumerate(action_columns):
            detail_data[f'actual_{col}'] = val_act_orig[:, i]
            detail_data[f'pred_{col}'] = pred_actions_orig[:, i]
            detail_data[f'constrained_{col}'] = constrained_actions[:, i]
        detail_data['reward'] = val_rew

        detail_df = pd.DataFrame(detail_data)
        detail_path = output_dir / "eval_predictions.csv"
        detail_df.to_csv(detail_path, index=False)
        logger.info(f"예측 상세 저장: {detail_path}")

    logger.info("=" * 80)
    logger.info("평가 완료")
    logger.info("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CQL 모델 평가 (validation 데이터)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 평가
  python -m apc_optimization.cql.evaluate

  # 경로 지정
  python -m apc_optimization.cql.evaluate \\
      --model-path outputs/cql/training/models/best_model.d3 \\
      --scaler-path outputs/cql/training/scaler.pkl \\
      --data-file outputs/offline_rl_training_data.parquet

  # 결과 저장
  python -m apc_optimization.cql.evaluate --output-dir outputs/cql/eval_results
        """
    )

    parser.add_argument(
        '--model-path', type=str, default=None,
        help='학습된 모델 경로 (.d3) (기본: outputs/cql/training/models/best_model.d3)'
    )
    parser.add_argument(
        '--scaler-path', type=str, default=None,
        help='스케일러 경로 (.pkl) (기본: outputs/cql/training/scaler.pkl)'
    )
    parser.add_argument(
        '--data-file', type=str, default=None,
        help='MDP 데이터 파일 (기본: outputs/offline_rl_training_data.parquet)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='결과 저장 디렉토리 (지정하지 않으면 콘솔 출력만)'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42,
        help='랜덤 시드 (학습 시와 동일해야 같은 val split 재현)'
    )

    args = parser.parse_args()

    # 기본 경로 설정
    training_dir = CQL_OUTPUT_DIR / "training"

    model_path = args.model_path or str(training_dir / "models" / "best_model.d3")
    scaler_path = args.scaler_path or str(training_dir / "scaler.pkl")
    data_file = args.data_file or str(PROJECT_ROOT / "outputs" / "offline_rl_training_data.parquet")

    # 파일 존재 확인
    for label, path in [("모델", model_path), ("스케일러", scaler_path), ("데이터", data_file)]:
        if not Path(path).exists():
            print(f"ERROR: {label} 파일을 찾을 수 없습니다: {path}")
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else (CQL_OUTPUT_DIR / "eval_results")

    print("=" * 80)
    print("CQL 모델 평가")
    print("=" * 80)
    print(f"모델:     {model_path}")
    print(f"스케일러: {scaler_path}")
    print(f"데이터:   {data_file}")
    print(f"출력:     {output_dir}")
    print("=" * 80)
    print()

    results = evaluate(
        model_path=model_path,
        scaler_path=scaler_path,
        data_file=data_file,
        output_dir=output_dir,
        random_seed=args.random_seed,
    )

    # 최종 요약
    print()
    print("=" * 80)
    print("평가 결과 요약")
    print("=" * 80)
    print(f"Validation 샘플: {results['n_val_samples']}")
    print(f"[정규화 공간] MSE: {results['normalized_space']['mse']:.6f}, MAE: {results['normalized_space']['mae']:.6f}")
    print(f"[원본 스케일] MSE: {results['original_scale']['mse']:.6f}, MAE: {results['original_scale']['mae']:.6f}")
    print(f"[제약 적용]   MSE: {results['constrained']['mse']:.6f}, MAE: {results['constrained']['mae']:.6f}")
    print(f"제약 위반율: {results['constrained']['violation_rate_pct']:.1f}%")
    print(f"평균 reward: {results['reward_stats']['mean']:.4f}")
    print("=" * 80)

    if output_dir:
        print(f"\n결과 저장 위치: {output_dir}")
        print("  - eval_results.json: 평가 메트릭")
        print("  - eval_predictions.csv: 예측 vs 실제 상세")

    return 0


if __name__ == "__main__":
    exit(main())
