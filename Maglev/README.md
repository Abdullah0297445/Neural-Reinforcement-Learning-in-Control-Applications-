This project has two milestones/parts: 

1. We created a [magnetic levitation](#magnetic-levitation) gym environment.
2. We have interfaced that environment with a neural network which learns to control it without any knowledge of the system itself.

# A little background

## Magnetic Levitation 

Magnetic levitation is a suspension mechanism in which an electromagnet can pull a metallic object vertically due to the magnetic force. This magnetic force is proportional to the current applied on the electromagnet and inversely proportional to the square of the distance between them. So it becomes a non-linear problem.

<img src="https://live.staticflickr.com/1425/667252542_4ec1625e90_b.jpg" width="400" height="400" />

## Gym Environment
A gym environment is a computer model (modelling of real world physics) of any real world problem.
It allows us to write general reinforcement learning algorithms to interface with it.

## Reinforcement Learning
It is a type of Artificial intelligence in which there is no direct supervision involved. 
Our agent learns the dynamics/inner workings of a system only by the interacting with it. 
The reward or punishment determined by the environment for every action our agent performs on it controls the quality of learning.

# Technical details

We have used Python 3.6+ for this project. 
Our environment is built in OpenAI's gym framework for Python. 
For deep learning, we have used Pytorch which is a fairly new framework because it is very easy to understand and debug.
