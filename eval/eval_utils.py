import re
import numpy as np

def extract_choice(text):
    """从回答中提取选项字母 (A/B/C/D等)"""
    text = str(text).strip()
    # 匹配选项模式
    patterns = [
        r'^([A-Z])[.、:：)]',  # A. 或 A、 或 A: 等
        r'[选择答案是]?\s*([A-Z])\s*[.、:：)]?$',  # 结尾的选项,
        # r'[the correct answer is:]?\s*([A-Z])\s*[.、:：)]?$',  # 结尾的选项
        r'^\s*([A-Z])\s*$',  # 单独的字母
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text.upper() if len(text) == 1 and text.isalpha() else text


def calculate_choice_accuracy(generated, gt):
    """计算选择题准确率（提取选项后比较）"""
    correct = 0
    for g, t in zip(generated, gt):
        pred_choice = extract_choice(g)
        true_choice = extract_choice(t)
        if pred_choice == true_choice:
            correct += 1
    return correct / len(generated) if generated else 0.0

def normalize_answer(s):
    """标准化答案文本"""
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s).lower().strip()
    # 移除标点符号
    s = re.sub(r'[^\w\s]', '', s)
    # 移除多余空格
    s = ' '.join(s.split())
    return s

def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_relaxed_accuracy(generated, gt):
    """计算宽松准确率（答案包含在生成文本中即可）"""
    correct = 0
    for g, t in zip(generated, gt):
        g_norm = normalize_answer(g)
        # 处理ground truth是列表的情况（如VizWiz）
        if isinstance(t, list):
            t_list = [normalize_answer(ans) for ans in t]
            if any(ans in g_norm or g_norm in ans for ans in t_list if ans):
                correct += 1
        else:
            t_norm = normalize_answer(t)
            if t_norm in g_norm or g_norm in t_norm:
                correct += 1
    return correct / len(generated) if generated else 0.0

def calculate_anls(generated, gt, threshold=0.5):
    """
    计算ANLS (Average Normalized Levenshtein Similarity)
    常用于OCR和文档理解任务
    """
    def normalized_levenshtein(s1, s2):
        s1, s2 = normalize_answer(s1), normalize_answer(s2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = levenshtein_distance(s1, s2)
        similarity = 1.0 - distance / max(len(s1), len(s2))
        return similarity if similarity >= threshold else 0.0
    
    scores = []
    for g, t in zip(generated, gt):
        if isinstance(t, list):
            # 取与任一正确答案的最高分
            score = max(normalized_levenshtein(g, ans) for ans in t)
        else:
            score = normalized_levenshtein(g, t)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def calculate_yes_no_accuracy(generated, gt):
    """计算Yes/No问答准确率"""
    correct = 0
    for g, t in zip(generated, gt):
        g_norm = normalize_answer(g)
        t_norm = normalize_answer(t)
        
        # 提取yes/no
        g_answer = 'yes' if 'yes' in g_norm else ('no' if 'no' in g_norm else g_norm)
        t_answer = 'yes' if 'yes' in t_norm else ('no' if 'no' in t_norm else t_norm)
        
        if g_answer == t_answer:
            correct += 1
    
    return correct / len(generated) if generated else 0.0


# 使用示例
if __name__ == "__main__":
    # 选择题示例
    gen_choice = ["A", "The answer is B", "C", "D"]
    gt_choice = ["A", "B", "C", "D"]
    results = calculate_choice_accuracy(gen_choice, gt_choice)
    print(f"Choice Accuracy: {results:.2%}")
    
    # 简短问答示例
    gen_short = ["Paris", "The capital is London", "42", "red"]
    gt_short = ["Paris", "London", "42", "Red"]
    anls = calculate_anls(gen_short, gt_short)
    results = calculate_relaxed_accuracy(gen_short, gt_short)
    print(f"ANLS: {anls:.4f}")
    print(f"Relaxed Accuracy: {results:.2%}")
    
    # VizWiz多答案示例
    gen_vizwiz = ["The image shows a cat", "This is a dog."]
    gt_vizwiz = [["cat", "kitten", "feline"], ['Cat', 'Dick']]
    anls = calculate_anls(gen_vizwiz, gt_vizwiz)
    results = calculate_relaxed_accuracy(gen_vizwiz, gt_vizwiz)
    print(f"ANLS: {anls:.4f}")
    print(f"Relaxed Accuracy: {results:.2%}")
    
    # Yes/No问答示例
    gen_yesno = ["Yes", "No, it is not", "Yes", "No"]
    gt_yesno = ["Yes", "No", "Yes", "No"]
    results = calculate_yes_no_accuracy(gen_yesno, gt_yesno)
    print(f"Yes/No Accuracy: {results:.2%}")
