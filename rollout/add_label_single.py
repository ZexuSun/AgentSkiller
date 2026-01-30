import os
import json
import copy
from pathlib import Path

# 假设的环境变量，请根据实际情况修改
POLICY_ROOT = "rollout/tools/datasets/single_domain/policies"  
TOOLS_LIST = "rollout/tools/datasets/single_domain/tool_lists"
OUTPUT_DIR = "outputs_single"
RESULT_FILE = "./mt_single_domain_tool_call_thinking.jsonl"

def main():
    with open(RESULT_FILE, "w", encoding="utf-8") as f_out:
        # 遍历输出目录
        if not os.path.exists(OUTPUT_DIR):
            print(f"Error: {OUTPUT_DIR} not found.")
            return

        for filename in os.listdir(OUTPUT_DIR):
            if not filename.endswith(".jsonl"): 
                continue
            
            # 2. 加载 Policy
            file_path = os.path.join(OUTPUT_DIR, filename)
            domain_name = filename.split(".")[0]
            policy_content = open(os.path.join(POLICY_ROOT, f"{domain_name}.md")).read()

            # 3. 加载 Tools
            tool_path = os.path.join(TOOLS_LIST, f"{domain_name}.json")
            tools = json.load(open(tool_path, "r", encoding="utf-8"))

            # 4. 处理每一行数据
            print(f"Processing {filename}...")
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line: continue
                    if "OpenAIException" in line: continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 插入 System Prompt
                        # 检查是否已经存在 system prompt，避免重复添加 (可选)
                        if data.get("messages") and data["messages"][0].get("role") != "system":
                            data["messages"].insert(0, {"role": "system", "content": policy_content})
                        
                        # 插入 Tools
                        data["tools"] = tools
                        
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {filename}")

if __name__ == "__main__":
    main()