{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████| 139M/139M [03:03<00:00, 793kiB/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "可以了请用100个字介绍你自己只100个字 为什么因为我就有这么长的时间听来面试听这么多我的时间有限 不懂吗可是你也应该知道正常人的语素一分钟是300个字那100个字大概是20秒钟左右我只用20秒钟的时间介绍自己你们确信能了解我那就看看技巧了问题不在我这儿那好开始我是一个最近有点倒霉的男人我需要一份专业对口的工作我可以随意见他我的尊业五年前我也曾经坐在你们这样的位置上对别人进行过面试五年后我仍然有可能坐在你的位置上对你进行面试如果我面试你我会给你充分的尊重基本的心上和友善的理解因为在这个机会和调查共存的时代任何成功和失败都是30的没有谁的成功是重生的也没有谁的失败是长久的如果你不认为我说的话有道理说明你这个人是个弱智如果你认为我说的话有道理那就行你给其他的面试者最起码的尊重你可以主宰一场面试可是你无法永远住在别人的命运我不认为贵公司这样的照片方法能够达到逛聚天下英才的目的万了正好20秒至于是不是100个资儿我就不知道了不过有一点我知道今天我来被你们选择纯粹吃一场乌驴一生过这么狂呢\n"
     ]
    }
   ],
   "source": [
    "import whisper\n",
    "\n",
    "model = whisper.load_model(\"base\")\n",
    "result = model.transcribe(\"C:/Users/USER/Desktop/audio.mp3\")\n",
    "print(result[\"text\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recording...\n",
      "Recording complete.\n",
      " you\n"
     ]
    }
   ],
   "source": [
    "import whisper\n",
    "import sounddevice as sd\n",
    "import tempfile\n",
    "import wave\n",
    "\n",
    "def record_audio(duration, samplerate=16000):\n",
    "    print(\"Recording...\")\n",
    "    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')\n",
    "    sd.wait()\n",
    "    print(\"Recording complete.\")\n",
    "    return audio, samplerate\n",
    "\n",
    "def save_audio_to_tempfile(audio, samplerate):\n",
    "    with tempfile.NamedTemporaryFile(delete=False, suffix=\".wav\") as temp_file:\n",
    "        with wave.open(temp_file.name, 'wb') as wf:\n",
    "            wf.setnchannels(1)\n",
    "            wf.setsampwidth(2)\n",
    "            wf.setframerate(samplerate)\n",
    "            wf.writeframes(audio.tobytes())\n",
    "        return temp_file.name\n",
    "\n",
    "model = whisper.load_model(\"base\")\n",
    "audio, samplerate = record_audio(10) #sec\n",
    "\n",
    "temp_audio_file = save_audio_to_tempfile(audio, samplerate)\n",
    "\n",
    "result = model.transcribe(temp_audio_file)\n",
    "print(result[\"text\"])"
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
