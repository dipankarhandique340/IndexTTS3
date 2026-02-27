#!/usr/bin/env python3
"""
IndexTTS2 CPU/Low-VRAM Test Suite
===================================
Tests the optimized inference engine to verify it works correctly.
"""

import os
import sys
import time
import argparse

def test_model_loading():
    """Test that all models load successfully."""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    from infer_cpu import IndexTTS2_CPU
    
    start = time.time()
    tts = IndexTTS2_CPU(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        device="cpu",
        use_quantization=True,
        verbose=True
    )
    elapsed = time.time() - start
    
    print(f"\n>> Model loading time: {elapsed:.1f}s")
    print(">> TEST 1 PASSED: All models loaded successfully!")
    return tts


def test_basic_inference(tts):
    """Test basic TTS inference with English text."""
    print("\n" + "=" * 60)
    print("TEST 2: Basic English Inference")
    print("=" * 60)
    
    # Find a voice sample
    voice_files = [
        "examples/voice_01.wav",
        "examples/voice_07.wav", 
        "examples/voice_10.wav",
    ]
    
    voice = None
    for vf in voice_files:
        if os.path.exists(vf):
            voice = vf
            break
    
    if voice is None:
        print(">> SKIPPED: No voice sample found in examples/")
        return False
    
    text = "Hello, this is a test of the IndexTTS2 model running on CPU."
    output_path = "test_output_cpu.wav"
    
    start = time.time()
    result = tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=output_path,
        verbose=True
    )
    elapsed = time.time() - start
    
    if result and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n>> Inference time: {elapsed:.1f}s")
        print(f">> Output file: {output_path} ({file_size} bytes)")
        print(">> TEST 2 PASSED: Basic inference works!")
        return True
    else:
        print(">> TEST 2 FAILED: No output generated")
        return False


def test_chinese_inference(tts):
    """Test Chinese text inference."""
    print("\n" + "=" * 60)
    print("TEST 3: Chinese Text Inference")
    print("=" * 60)
    
    voice_files = [
        "examples/voice_01.wav",
        "examples/voice_07.wav",
    ]
    
    voice = None
    for vf in voice_files:
        if os.path.exists(vf):
            voice = vf
            break
    
    if voice is None:
        print(">> SKIPPED: No voice sample found")
        return False
    
    text = "欢迎大家来体验 AI 科技。"
    output_path = "test_output_cpu_zh.wav"
    
    start = time.time()
    result = tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=output_path,
        verbose=True
    )
    elapsed = time.time() - start
    
    if result and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n>> Inference time: {elapsed:.1f}s")
        print(f">> Output file: {output_path} ({file_size} bytes)")
        print(">> TEST 3 PASSED: Chinese inference works!")
        return True
    else:
        print(">> TEST 3 FAILED: No output generated")
        return False


def test_emotion_vector(tts):
    """Test emotion vector control."""
    print("\n" + "=" * 60)
    print("TEST 4: Emotion Vector Control")
    print("=" * 60)
    
    voice_files = [
        "examples/voice_01.wav",
        "examples/voice_10.wav",
    ]
    
    voice = None
    for vf in voice_files:
        if os.path.exists(vf):
            voice = vf
            break
    
    if voice is None:
        print(">> SKIPPED: No voice sample found")
        return False
    
    text = "This is amazing! I can't believe how good this sounds!"
    output_path = "test_output_cpu_emo.wav"
    # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    emo_vector = [0.6, 0, 0, 0, 0, 0, 0.3, 0]
    
    start = time.time()
    result = tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=output_path,
        emo_vector=emo_vector,
        verbose=True
    )
    elapsed = time.time() - start
    
    if result and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n>> Inference time: {elapsed:.1f}s")
        print(f">> Output file: {output_path} ({file_size} bytes)")
        print(">> TEST 4 PASSED: Emotion vector control works!")
        return True
    else:
        print(">> TEST 4 FAILED: No output generated")
        return False


def test_memory_usage():
    """Report memory usage."""
    print("\n" + "=" * 60)
    print("TEST 5: Memory Usage Report")
    print("=" * 60)
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f">> RAM usage: {ram_mb:.0f} MB")
    except ImportError:
        print(">> psutil not installed, skipping RAM check")
    
    import torch
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f">> VRAM allocated: {vram_allocated:.0f} MB")
        print(f">> VRAM reserved: {vram_reserved:.0f} MB")
    else:
        print(">> Running on CPU (no VRAM usage)")
    
    print(">> TEST 5 PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test IndexTTS2 CPU optimization")
    parser.add_argument("--device", type=str, default="cpu", help="Device to test on")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference tests (only test loading)")
    args = parser.parse_args()

    print("=" * 60)
    print("IndexTTS2 CPU/Low-VRAM Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Loading
    try:
        tts = test_model_loading()
        results["loading"] = True
    except Exception as e:
        print(f">> TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["loading"] = False
        print("\nCannot continue without model loading. Exiting.")
        return
    
    if not args.skip_inference:
        # Test 2: Basic inference
        try:
            results["english"] = test_basic_inference(tts)
        except Exception as e:
            print(f">> TEST 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["english"] = False
        
        # Test 3: Chinese
        try:
            results["chinese"] = test_chinese_inference(tts)
        except Exception as e:
            print(f">> TEST 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["chinese"] = False
        
        # Test 4: Emotion
        try:
            results["emotion"] = test_emotion_vector(tts)
        except Exception as e:
            print(f">> TEST 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["emotion"] = False
    
    # Test 5: Memory
    results["memory"] = test_memory_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {name}: {'PASSED' if status else 'FAILED'}")
    print(f"\n  Result: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
