import os
import json
import copy
from pathlib import Path

# 假设的环境变量，请根据实际情况修改
POLICY_ROOT = "rollout/tools/datasets/cross_domain/policies"  
TOOLS_LIST = "rollout/tools/datasets/cross_domain/tool_lists"
OUTPUT_DIR = "outputs_short_thinking"
RESULT_FILE = "./mt_tool_call_thinking.jsonl"

def build_policy_map(policy_root):
    """
    预处理 Policy 目录。
    将文件名拆解并排序，生成一个 frozenset 作为 Key。
    这样 A_B_C.md 和 C_B_A.md (如果存在) 都会被映射到同一个集合上。
    """
    policy_map = {}
    if not os.path.exists(policy_root):
        print(f"Warning: {policy_root} does not exist.")
        return policy_map

    for filename in os.listdir(policy_root):
        if not filename.endswith(".md"):
            continue
        
        # 去掉 .md 后缀
        domain_str = filename[:-3] 
        # 将 "A_B_C" 拆分为 ["A", "B", "C"]
        parts = domain_str.split("_")
        # 使用 frozenset 忽略顺序: {"A", "B", "C"}
        # 这样无论你的文件名是 A_B.md 还是 B_A.md，Key 都是一样的
        key = frozenset(parts)
        
        full_path = os.path.join(policy_root, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            policy_map[key] = f.read()
            
    return policy_map

def main():
    # 1. 预加载所有 Policy，解决顺序问题
    print("Loading policies...")
    policy_map = build_policy_map(POLICY_ROOT)
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f_out:
        # 遍历输出目录
        if not os.path.exists(OUTPUT_DIR):
            print(f"Error: {OUTPUT_DIR} not found.")
            return

        for filename in os.listdir(OUTPUT_DIR):
            if not filename.endswith(".jsonl"): 
                continue

            file_path = os.path.join(OUTPUT_DIR, filename)
            # 获取文件名作为 domain 组合，例如 "finance_travel.jsonl" -> "finance_travel"
            domain_comb_name = filename.split(".")[0]
            
            # 2. 查找对应的 Policy
            # 将当前文件的 domain 组合也拆分并转为 set
            current_parts = domain_comb_name.split("_")
            current_key = frozenset(current_parts)
            
            policy_content = policy_map.get(current_key)
            
            if not policy_content:
                print(f"Warning: No matching policy found for {domain_comb_name} (Checked permutations)")
                # 可以选择 continue 跳过，或者使用一个默认 Policy
                policy_content = "Default Policy..." 

            # 3. 加载 Tools
            tools = []
            for single_domain in current_parts:
                tool_path = os.path.join(TOOLS_LIST, f"{single_domain}.json")
                if os.path.exists(tool_path):
                    with open(tool_path, "r", encoding="utf-8") as tf:
                        tools.extend(json.load(tf))
                else:
                    print(f"Warning: Tool file not found: {tool_path}")

            # 处理 Tool 参数 (转换成 string)
            processed_tools = []
            for tool in tools:
                # [重要] 使用 deepcopy，防止修改原始引用导致后续循环数据污染
                new_tool = copy.deepcopy(tool)
                if "function" in new_tool and "parameters" in new_tool["function"]:
                    # 这里是原来的逻辑：把 dict 转成 string
                    new_tool["function"]["parameters"] = json.dumps(
                        new_tool["function"]["parameters"], ensure_ascii=False
                    )
                processed_tools.append(new_tool)

            # 4. 处理每一行数据
            print(f"Processing {filename}...")
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 插入 System Prompt
                        # 检查是否已经存在 system prompt，避免重复添加 (可选)
                        if data.get("messages") and data["messages"][0].get("role") != "system":
                            data["messages"].insert(0, {"role": "system", "content": policy_content})
                        
                        # 插入 Tools
                        # [Bug修复] 原代码 data[tools] = tools 是错误的语法 (list不能做key)
                        data["tools"] = processed_tools
                        
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {filename}")

if __name__ == "__main__":
    main()