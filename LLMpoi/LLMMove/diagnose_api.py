"""诊断脚本 v2：只测 chat completion"""
import os, sys, json, time

API_BASE = os.environ.get("API_BASE", "https://opencode.ai/zen/go/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

print(f"API Base: {API_BASE}")
print(f"Model:    {MODEL}")
print(f"API Key:  {'[SET]' if API_KEY else '[EMPTY]'}")
print()

assert API_KEY, "❌ OPENAI_API_KEY 未设置"

from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=30, max_retries=0)

# Test 1: Simple chat completion
print("1. 测试 chat completion (简单请求, timeout=30s)...")
try:
    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say only: hello world"}],
        temperature=0,
    )
    content = response.choices[0].message.content
    print(f"   ✅ 成功 ({time.time()-t0:.1f}s)")
    print(f"   回复: {repr(content[:200])}")
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    
    # Test 2: Try without temperature
    print()
    print("2. 尝试不带 temperature...")
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say only: hello world"}],
        )
        content = response.choices[0].message.content
        print(f"   ✅ 成功 ({time.time()-t0:.1f}s)")
        print(f"   回复: {repr(content[:200])}")
    except Exception as e2:
        print(f"   ❌ 也失败: {type(e2).__name__}: {e2}")
        
        # Test 3: Try different model names
        for alt_model in ["deepseek-v4-flash", "deepseek-chat", "deepseek-ai/DeepSeek-V3"]:
            if alt_model == MODEL:
                continue
            print(f"\n3. 尝试模型 {alt_model}...")
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=alt_model,
                    messages=[{"role": "user", "content": "Say only: hello"}],
                )
                content = response.choices[0].message.content
                print(f"   ✅ 成功 ({time.time()-t0:.1f}s)")
                print(f"   模型: {alt_model}")
                print(f"   回复: {repr(content[:200])}")
                break
            except Exception as e3:
                print(f"   ❌ {type(e3).__name__}: {str(e3)[:100]}")
