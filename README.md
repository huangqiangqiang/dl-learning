# 

docker run -it -d --name tensorflow-gpu --gpus all -v $(pwd):/app -v /home/hqq/work/dataset:/dataset hqqsk8/tensorflow:2.4.1-gpu-jupyter 

# 查看 gpu 使用率

```
watch -n 1 nvidia-smi
```