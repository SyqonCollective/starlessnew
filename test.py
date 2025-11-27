"""
Test script to verify model and dataset
"""

import torch
import sys
from pathlib import Path

def test_model():
    """Test model creation and forward pass"""
    print("=" * 60)
    print("Testing Model...")
    print("=" * 60)
    
    try:
        from model import create_msrf_nafnet_s, create_msrf_nafnet_m, create_msrf_nafnet_l
        
        # Test small model
        print("\n1. Testing MSRF-NAFNet-S...")
        model = create_msrf_nafnet_s()
        x = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            y = model(x)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created successfully")
        print(f"   ✓ Parameters: {total_params:,}")
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {y.shape}")
        assert y.shape == x.shape, "Output shape mismatch!"
        
        # Test medium model
        print("\n2. Testing MSRF-NAFNet-M...")
        model = create_msrf_nafnet_m()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Parameters: {total_params:,}")
        
        # Test large model
        print("\n3. Testing MSRF-NAFNet-L...")
        model = create_msrf_nafnet_l()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Parameters: {total_params:,}")
        
        print("\n✅ Model tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading"""
    print("\n" + "=" * 60)
    print("Testing Dataset...")
    print("=" * 60)
    
    try:
        from dataset import StarRemovalDataset
        
        root_dir = "/Users/michaelruggeri/Desktop/starless"
        
        # Check if directories exist
        train_input = Path(root_dir) / "train" / "input"
        train_target = Path(root_dir) / "train" / "target"
        
        if not train_input.exists():
            print(f"   ⚠ Warning: {train_input} does not exist")
            print(f"   Please ensure your dataset is organized as:")
            print(f"   {root_dir}/")
            print(f"   ├── train/")
            print(f"   │   ├── input/")
            print(f"   │   └── target/")
            print(f"   └── val/")
            print(f"       ├── input/")
            print(f"       └── target/")
            return False
        
        print(f"\n1. Creating training dataset...")
        train_dataset = StarRemovalDataset(
            root_dir=root_dir,
            split='train',
            patch_size=256
        )
        print(f"   ✓ Train dataset size: {len(train_dataset)}")
        
        print(f"\n2. Loading sample...")
        if len(train_dataset) > 0:
            input_img, target_img = train_dataset[0]
            print(f"   ✓ Input shape: {input_img.shape}")
            print(f"   ✓ Target shape: {target_img.shape}")
            print(f"   ✓ Input range: [{input_img.min():.3f}, {input_img.max():.3f}]")
            print(f"   ✓ Target range: [{target_img.min():.3f}, {target_img.max():.3f}]")
        else:
            print("   ⚠ Warning: No images found in dataset")
            return False
        
        print(f"\n3. Creating validation dataset...")
        val_dataset = StarRemovalDataset(
            root_dir=root_dir,
            split='val',
            patch_size=256,
            augment=False
        )
        print(f"   ✓ Val dataset size: {len(val_dataset)}")
        
        print("\n✅ Dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses():
    """Test loss functions"""
    print("\n" + "=" * 60)
    print("Testing Loss Functions...")
    print("=" * 60)
    
    try:
        from losses import CombinedLoss
        
        print("\n1. Creating loss function...")
        criterion = CombinedLoss()
        print("   ✓ Loss function created")
        
        print("\n2. Testing forward pass...")
        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        
        loss, components = criterion(pred, target, return_components=True)
        
        print("   ✓ Loss components:")
        for name, value in components.items():
            print(f"      {name}: {value:.6f}")
        
        print("\n✅ Loss tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing CUDA...")
    print("=" * 60)
    
    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"3. CUDA version: {torch.version.cuda}")
        print(f"4. Device count: {torch.cuda.device_count()}")
        print(f"5. Device name: {torch.cuda.get_device_name(0)}")
        print(f"6. Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU computation
        print(f"\n7. Testing GPU computation...")
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print(f"   ✓ GPU computation successful")
        
        # Test mixed precision
        print(f"\n8. Testing mixed precision...")
        from torch.cuda.amp import autocast
        with autocast():
            z = torch.mm(x, y)
        print(f"   ✓ Mixed precision successful")
        
        print("\n✅ CUDA tests passed!")
    else:
        print("\n⚠ Warning: CUDA not available. Training will use CPU (slow)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MSRF-NAFNet Test Suite")
    print("=" * 60)
    
    results = {
        "CUDA": test_cuda(),
        "Model": test_model(),
        "Losses": test_losses(),
        "Dataset": test_dataset(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:12s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
        print("\nQuick start:")
        print("  python train.py --config config.yaml")
    else:
        print("\n⚠ Some tests failed. Please fix issues before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
