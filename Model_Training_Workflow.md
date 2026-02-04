# 本地开发 + 云端训练 工作流（GitHub 代码 + Drive 数据）

本工作流实现“存算分离”：
- 代码（Git）在本地 E 盘开发并推送到 GitHub，Colab 从 GitHub 拉取最新代码。
- 数据与训练产物在 Google Drive（G 盘）存放；Colab 读取/写入 Drive。
- 训练完成后，用本地同步脚本将 Drive 上的指标/报告回传到仓库的 `reports/`，作为后续调参依据。

---

## 一、目录与职责

- 代码仓库（本地）：`E:\Projects\Graduation_Project`（Git 管理，不放大文件）。
- 远端代码：GitHub 仓库（Colab 端 `git clone`/`git pull`）。
- 数据与产物（云端）：`/content/drive/MyDrive/Graduation_Project` 下：
  - 数据：`data/processed/...`
  - 实验：`experiments/<run_name>/...`（训练输出目录 `--output_dir` 指向此处）
- 结果回传（本地）：`reports/<run_name>/...`（由同步脚本生成，仅小文件）

---

## 二、Colab 端：最小 Runner

1) 准备环境（每次新会话建议先升级依赖）
```python
!pip install -U transformers datasets evaluate huggingface_hub
```

2) 挂载 Drive（用于数据与产物）
```python
from google.colab import drive
drive.mount('/content/drive')
```

3) 拉取代码（GitHub）并进入仓库目录
```bash
!git clone https://github.com/<your_org>/<your_repo>.git  # 首次
%cd /content/<your_repo>
# 之后更新用：!git -C /content/<your_repo> pull
```

4) 运行训练（将输出目录指向 Drive）
```bash
python scripts/modeling/bert_finetune_cls.py \
  --train_csv /content/drive/MyDrive/Graduation_Project/data/processed/train.csv \
  --val_csv   /content/drive/MyDrive/Graduation_Project/data/processed/val.csv \
  --test_csv  /content/drive/MyDrive/Graduation_Project/data/processed/test.csv \
  --output_dir /content/drive/MyDrive/Graduation_Project/experiments/bert_6cls_YYYYMMDDA \
  --model_name hfl/chinese-roberta-wwm-ext \
  --epochs 5 --lr 1e-5 --max_length 384 \
  --class_weight auto --warmup_ratio 0.06 --weight_decay 0.01 \
  --eval_steps 100 --save_steps 100 --early_stopping_patience 2
```

说明：
- 训练脚本已兼容旧版 transformers 的 EarlyStopping 与 Trainer 参数差异，并会自动写出 `eval_results.json/metrics_*.json/report_test.txt/pred_test.csv/best/config.json`。
- 403 “Discussions disabled” 日志可忽略；不影响加载与训练。

---

## 三、本地端：同步训练产物为报告

运行同步脚本，仅复制小文件到仓库 `reports/`（默认跳过大权重）：
```powershell
python scripts/tools/sync_results.py `
  --src_root "G:\我的云端硬盘\Graduation_Project\experiments" `
  --dst_root "E:\Projects\Graduation_Project\reports" `
  --dry_run --verbose    # 先预演

# 实际执行
python scripts/tools/sync_results.py --verbose
```

可通过 include/exclude 精细控制（示例：加入最优权重）：
```powershell
python scripts/tools/sync_results.py `
  --include "**/eval_results.json" --include "**/metrics*.json" `
  --include "**/report*.txt" --include "**/pred*.csv" `
  --include "**/best/config.json" --include "**/*.safetensors"
```

---

## 四、常见问答（FAQ）

- 是否必须创建/上传 .ipynb？
  - 否。推荐在 Colab 新建一个极简 Runner 笔记本，仅包含本页“二、Colab 端：最小 Runner”的几段单元即可；也可直接在 Colab 的终端执行上述命令。
  - 不需要把现有 `notebooks/bert_multilabel_colab.ipynb` 上传到 Drive；如需可在 Colab 的“从 GitHub 打开”中直接打开它。

- 训练数据放哪里？
  - 放在 `/content/drive/MyDrive/Graduation_Project/data/processed/...`，训练命令直接指向该路径。

- 训练输出放哪里？
  - 指定 `--output_dir` 到 `/content/drive/MyDrive/Graduation_Project/experiments/<run_name>`。

- 本地如何获取指标？
  - 运行 `scripts/tools/sync_results.py`，会将关键小文件复制到 `reports/<run_name>/`。

---

## 五、调参与改进建议

- 类不平衡：使用 `--class_weight auto`（已内置权重计算）。
- 训练稳定性：启用 EarlyStopping（已修复断言与兼容旧版本）、`warmup_ratio`、`weight_decay`、step 级评估与保存。
- 文本长度：中文长文本建议 `--max_length 384~512`；必要时裁剪/清洗噪声。
- 进一步方向：focal loss、分层标签或后处理阈值（按需要追加到脚本）。
