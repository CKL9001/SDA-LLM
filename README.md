# SDA-LLM: Spatial DisAmbiguation via Multi-turn Vision-Language Dialogues for Robot Navigation

## 📢 Welcome to Our Paper!

Thank you for your interest in our research project —  
**SDA-LLM: Spatial DisAmbiguation via Multi-turn Vision-Language Dialogues for Robot Navigation**.

This work addresses a common challenge in human-robot interaction: when users give natural language instructions involving **relative positional references**, such as _“Go to the chair and pick up empty bottles,”_ ambiguity often arises. This is especially problematic when:

- Multiple similar objects (e.g., several chairs) exist in the environment, or  
- The robot’s field of view is limited, making it difficult to interpret the instruction precisely.

To tackle this problem, we propose a **two-level framework** that combines a **Large Language Model (LLM)** with a **Vision-Language Model (VLM)**. Our approach enables the robot to engage in **multi-turn dialogues** with the user to clarify spatial references and identify the correct navigation target.

### 🔍 Key Features

- 🧠 Maps dialogue semantics to a **unique object ID** in the image using a VLM  
- 🌐 Connects the object ID to a **3D depth map** for accurate spatial localization  
- 🤖 Enables **interactive, multi-turn clarification** between user and robot

To the best of our knowledge, this is the **first work to leverage foundation models for spatial disambiguation** in robot navigation.

---

We warmly invite you to explore our paper, code, and approach.  
Your feedback and collaboration are most welcome!

📄 Please view the [full paper](https://arxiv.org/abs/2410.12802) and [video](https://www.youtube.com/watch?v=_CxhU5LAYLw).
