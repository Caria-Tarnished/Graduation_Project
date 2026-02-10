# Google Drive 同步大文件问题解决方案

## 问题描述
`reports/bert_3cls_enhanced_v1/best/model.safetensors` (390MB) 不断重复上传，占用大量网络带宽。

## 解决方案

### 方法 1：从 Google Drive 同步中排除 reports 目录（推荐）

1. **右键点击 Google Drive 托盘图标**（任务栏右下角）
2. **点击"设置"（齿轮图标）→ "首选项"**
3. **选择"Google Drive"标签页**
4. **点击"文件夹"旁边的"管理"按钮**
5. **找到你的项目文件夹，展开它**
6. **取消勾选 `reports` 文件夹**（或者只取消勾选 `reports/bert_3cls_enhanced_v1/best`）
7. **点击"保存"**

这样 Google Drive 就不会同步 reports 目录中的大型模型文件了。

### 方法 2：将文件移动到 Google Drive 外部

如果你不需要在 Google Drive 中备份模型文件：

```powershell
# 创建本地模型存储目录（不在 Google Drive 中）
mkdir F:\local_models\bert_3cls_enhanced_v1\best

# 移动模型文件
move reports\bert_3cls_enhanced_v1\best\model.safetensors F:\local_models\bert_3cls_enhanced_v1\best\

# 创建符号链接（可选，如果代码需要访问）
# 需要管理员权限
# mklink reports\bert_3cls_enhanced_v1\best\model.safetensors F:\local_models\bert_3cls_enhanced_v1\best\model.safetensors
```

### 方法 3：暂停 Google Drive 同步

如果你需要立即停止上传：

1. **右键点击 Google Drive 托盘图标**
2. **点击"暂停"**
3. **选择暂停时长**（例如"暂停 1 小时"或"暂停直到我恢复"）

然后按照方法 1 排除 reports 目录后，再恢复同步。

## 关于数据库文件上传

由于 `finance_analysis.db` (253MB) 也很大，建议：

### 选项 A：使用模板数据训练（推荐）
- Colab 训练代码已支持降级策略
- 使用 120 条指令数据
- 训练时间 30-60 分钟
- 足够演示 QLoRA 微调效果

### 选项 B：分时段上传数据库
1. 暂停 Google Drive 同步
2. 等待网络空闲时段（例如深夜）
3. 手动将 `finance_analysis.db` 复制到 Google Drive 文件夹
4. 恢复同步，让它慢慢上传

### 选项 C：使用 Google Drive 网页版上传
1. 访问 https://drive.google.com
2. 找到你的项目文件夹
3. 直接拖拽上传 `finance_analysis.db`
4. 网页版上传通常更稳定，支持断点续传

## 已更新的 .gitignore 规则

已添加以下规则防止 Git 跟踪大型模型文件：
```
*.safetensors
*.bin
*.ckpt
*.pth
*.pt
```

## 验证

运行以下命令确认文件未被 Git 跟踪：
```powershell
git status
```

应该看不到 `model.safetensors` 文件。
