# 

docker run -it -d --name tensorflow-gpu --gpus all -v $(pwd):/app -v /home/hqq/work/dataset:/dataset hqqsk8/tensorflow:2.4.1-gpu-jupyter 

# 查看 gpu 使用率

```
watch -n 1 nvidia-smi
```

# guanyu ImageDataGenerator de 
imagedatagenerator 不会增加训练集的大小。所有的扩充都是在内存中完成的。所以原始图像被随机增强，然后它的增强版本被返回。
```

```