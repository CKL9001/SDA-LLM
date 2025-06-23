{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 640x640 1 cabinet, 1 table, 10 chairs, 1 whiteboard, 6.0ms\n",
      "Speed: 6.0ms preprocess, 6.0ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 640)\n",
      "xmin: 907 ymin: 430 xmax: 1051 ymax: 644\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "from ultralytics import YOLO\n",
    "from PIL import Image\n",
    "\n",
    "model = YOLO(\"C:/Users/USER/Desktop/研/模組課(下)/ROSLLMdataset/runs/detect/train/weights/best.pt\")\n",
    "\n",
    "frame = Image.open(\"C:/Users/USER/Desktop/1.jpg\")\n",
    "frame = np.array(frame)\n",
    "frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)\n",
    "\n",
    "results = model(frame)\n",
    "for i, result in enumerate(results):\n",
    "    boxes = result.boxes\n",
    "    for j, box in enumerate(boxes):\n",
    "        class_id = box.cls.item()\n",
    "        if class_id == 0.0:\n",
    "            xyxy = box.xyxy.cpu().numpy()[0]\n",
    "            xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])\n",
    "            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)\n",
    "            print(\"xmin:\", xmin, \"ymin:\", ymin, \"xmax:\", xmax, \"ymax:\", ymax)\n",
    "\n",
    "cv2.imshow(\"Frame\", frame)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 1024x1024 323.2ms\n",
      "Speed: 10.0ms preprocess, 323.2ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)\n",
      "Results saved to \u001b[1mruns\\segment\\predict7\u001b[0m\n",
      "[[[0 0 0 ... 0 0 0]\n",
      "  [0 0 0 ... 0 0 0]\n",
      "  [0 0 0 ... 0 0 0]\n",
      "  ...\n",
      "  [0 0 0 ... 0 0 0]\n",
      "  [0 0 0 ... 0 0 0]\n",
      "  [0 0 0 ... 0 0 0]]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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dco59cqQfec73zH33ntv2LFLL73ULF68uJtW1HWqq6uNJFNUVGSMMaatrc34/X7zyCOPODOfffaZ8Xq95oknnjDGGPPpp5+a6Ohos2XLFmfmxIkTpl+/fubFF1/s2hOIkLq6OpORkWEKCwvNmDFjnJD1tf1YtGiRGT169Bmf72v7YYwxN910k/nxj38cduy2224zd911lzGmb+1J+5BF6twPHz5sJJndu3c7M7t27TKSzDvvvBPx8+h1f7XY1NSk0tJS5ebmhh3Pzc1VSUlJN62q64RCIUn/+dv/y8vLVVVVFbYfHo9HY8aMcfajtLRUzc3NYTOBQECZmZnW7tl9992nm266SePGjQs73tf244UXXtCIESN0++23KzU1VcOHD9eTTz7pPN/X9kOSRo8erZdeeklHjhyRJL355psqLi7WjTfeKKlv7skXInXuu3btktfrVXZ2tjPz3e9+V16vt1P2p9f99vuPP/5Yra2t8vl8Ycd9Pp+qqqq6aVVdwxijefPmafTo0crMzJQk55y/bD+OHj3qzMTExGjAgAEdZmzcsy1btmjfvn3au3dvh+f62n68//77Wrt2rebNm6ef/exn2rNnj+6//355PB796Ec/6nP7IUmLFi1SKBTSpZdeqqioKLW2turhhx/WnXfeKanv/Rn5W5E696qqKqWmpnZ4/dTU1E7Zn14Xsi+4XK6wr40xHY71NrNnz9Zbb72l4uLiDs/9Pfth455VVFRo7ty52rFjh/r373/Gub6yH21tbRoxYoTy8/MlScOHD9ehQ4e0du1a/ehHP3Lm+sp+SNKzzz6rTZs2afPmzbriiitUVlamYDCoQCCg6dOnO3N9aU/ai8S5f9l8Z+1Pr/urxZSUFEVFRXWofnV1dYf/l9GbzJkzRy+88IJeeeUVDRo0yDnu9/sl6az74ff71dTUpJqamjPO2KK0tFTV1dXKysqS2+2W2+1WUVGRfv3rX8vtdjvn01f246KLLtLll18eduyyyy7TsWPHJPW9Px+S9OCDD2rx4sW64447NGzYME2bNk0PPPCACgoKJPXNPflCpM7d7/frww8/7PD6H330UafsT68LWUxMjLKyslRYWBh2vLCwUKNGjeqmVXUeY4xmz56trVu36uWXX1Z6enrY8+np6fL7/WH70dTUpKKiImc/srKyFB0dHTZTWVmpgwcPWrdnY8eO1YEDB1RWVuY8RowYoalTp6qsrExDhw7tU/tx3XXXdfg4xpEjRzRkyBBJfe/PhySdPn3a+Yd3vxAVFeXcft8X9+QLkTr3kSNHKhQKac+ePc7MG2+8oVAo1Dn7E/HbR3qAL26/X79+vTl8+LAJBoMmPj7e/PWvf+3upUXcT3/6U+P1es2rr75qKisrncfp06edmUceecR4vV6zdetWc+DAAXPnnXd+6e20gwYNMjt37jT79u0zP/jBD6y4lfir+Nu7Fo3pW/uxZ88e43a7zcMPP2z+8pe/mGeeecbExcWZTZs2OTN9aT+MMWb69Olm4MCBzu33W7duNSkpKWbhwoXOTG/ek7q6OrN//36zf/9+I8msWrXK7N+/3/l4UqTO/YYbbjBXXnml2bVrl9m1a5cZNmwYt9+fr9/85jdmyJAhJiYmxlxzzTXO7ei9jaQvfTz11FPOTFtbm/nFL35h/H6/8Xg85vvf/745cOBA2Os0NDSY2bNnm6SkJBMbG2vy8vLMsWPHuvhsOkf7kPW1/fi3f/s3k5mZaTwej7n00kvNunXrwp7va/tRW1tr5s6dawYPHmz69+9vhg4dapYuXWoaGxudmd68J6+88sqX/jdj+vTpxpjInfvJkyfN1KlTTUJCgklISDBTp041NTU1nXJO/HtkAACr9bqfkQEA+hZCBgCwGiEDAFiNkAEArEbIAABWI2QAAKsRMgCA1QgZAMBqhAwAYDVCBgCwGiEDAFjt/wMkr5j+G7nt+QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from ultralytics import SAM\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "model = SAM(\"sam_b.pt\")\n",
    "result = model(frame, bboxes=[xmin, ymin, xmax, ymax], save = True)\n",
    "mask = (result[0].masks.data).cpu().numpy()\n",
    "plt.imshow(mask[0])\n",
    "\n",
    "int_mask = mask.astype(np.uint8)\n",
    "print(int_mask)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "kl",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
