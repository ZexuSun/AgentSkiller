import json
import hashlib

def generate_id(entry):
    return hashlib.md5(json.dumps(entry['messages'], sort_keys=True).encode("utf-8")).hexdigest()


with open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/mix.jsonl", "w") as f:
    for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/code_math_43000.jsonl"):
        try:
            data = json.loads(line)
            messages = [
                {
                    "role": msg.get("role", None),
                    "content": msg.get("content", None),
                    "reasoning_content": msg.get("reasoning_content", None),
                    "tool_calls": msg.get("tool_calls", None),
                    "tool_call_id": msg.get("tool_call_id", None)
                }
                for msg in data["messages"]
            ]
            new_data = {
                "messages": messages,
                "id": data.get("id", generate_id(data)),
                "data_source": "general",
                "use_cot": False,
                "tools": data.get("tools", None)
            }
            f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(e)

    for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/mt_single_domain_tool_call_thinking_splitted_formatted.jsonl"):
        try:
            data = json.loads(line)
            messages = [
                {
                    "role": msg.get("role", None),
                    "content": msg.get("content", None),
                    "reasoning_content": msg.get("reasoning_content", None),
                    "tool_calls": msg.get("tool_calls", None),
                    "tool_call_id": msg.get("tool_call_id", None)
                }
                for msg in data["messages"]
            ]
            new_data = {
                "messages": messages,
                "id": data.get("id", generate_id(data)),
                "data_source": "agent",
                "use_cot": True,
                "tools": data.get("tools", None)
            }
            f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(e)

    # for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/mt_tool_call_thinking_stopped_splited.jsonl"):
    #     data = json.loads(line)
    #     messages = [
    #         {
    #             "role": msg.get("role", None),
    #             "content": msg.get("content", None),
    #             "reasoning_content": msg.get("reasoning_content", None),
    #             "tool_calls": msg.get("tool_calls", None),
    #             "tool_call_id": msg.get("tool_call_id", None)
    #         }
    #         for msg in data["messages"]
    #     ]
    #     new_data = {
    #         "messages": messages,
    #         "id": data["id"],
    #         "data_source": "agent",
    #         "use_cot": True,
    #         "tools": data.get("tools", None)
    #     }
    #     f.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    # for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/mt_tool_call_thinking_transferred.jsonl"):
    #     data = json.loads(line)
    #     messages = [
    #         {
    #             "role": msg.get("role", None),
    #             "content": msg.get("content", None),
    #             "reasoning_content": msg.get("reasoning_content", None),
    #             "tool_calls": msg.get("tool_calls", None),
    #             "tool_call_id": msg.get("tool_call_id", None)
    #         }
    #         for msg in data["messages"]
    #     ]
    #     new_data = {
    #         "messages": messages,
    #         "id": data["id"],
    #         "data_source": "agent",
    #         "use_cot": True,
    #         "tools": data.get("tools", None)
    #     }
    #     f.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    # for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/swe_sample_3w.jsonl"):
    #     data = json.loads(line)
    #     messages = [
    #         {
    #             "role": msg.get("role", None),
    #             "content": msg.get("content", None),
    #             "reasoning_content": msg.get("reasoning_content", None),
    #             "tool_calls": msg.get("tool_calls", None),
    #             "tool_call_id": msg.get("tool_call_id", None)
    #         }
    #         for msg in data["messages"]
    #     ]
    #     new_data = {
    #         "messages": messages,
    #         "id": generate_id(data),
    #         "data_source": "agent",
    #         "use_cot": False,
    #         "tools": data.get("tools", None)
    #     }
    #     f.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    # for line in open("/root/paddlejob/workspace/env_run/output/rollout/Rollout/general_10000.jsonl"):
    #     data = json.loads(line)
    #     messages = [
    #         {
    #             "role": msg.get("role", None),
    #             "content": msg.get("content", None),
    #             "reasoning_content": msg.get("reasoning_content", None),
    #             "tool_calls": msg.get("tool_calls", None),
    #             "tool_call_id": msg.get("tool_call_id", None)
    #         }
    #         for msg in data["messages"]
    #     ]
    #     new_data = {
    #         "messages": messages,
    #         "id": generate_id(data),
    #         "data_source": "general",
    #         "use_cot": True,
    #         "tools": data.get("tools", None)
    #     }
    #     f.write(json.dumps(new_data, ensure_ascii=False) + "\n")