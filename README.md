# Multi-Agent-Simulation-Tool


# Multi-Agent Simulation Tool

This repository provides an **Advanced Multi-Agent Simulation Tool** designed to simulate various environments like economics, traffic, and robotics with both reinforcement learning (RL) and supervised agents. The tool is built using **Streamlit** for the user interface and **PyTorch** for machine learning models.

## Features

- **Multi-Agent Environments:**
  - Economic, Traffic, and Robotics environments.
  - Each environment has customizable parameters, such as the number of agents and maximum steps.

- **Agents:**
  - **Reinforcement Learning (RL) Agents:** Implemented using a Deep Q-Network (DQN) architecture.
  - **Supervised Agents:** Trained with supervised learning techniques.

- **Visualization:**
  - Total reward per episode.
  - Action heatmaps, t-SNE visualizations, and cluster analysis of agent behaviors.
  - Network analysis for agent interactions.

- **Google Gemini Integration:**
  - Provides AI-generated insights and explanations of agent strategies.

- **Performance Analysis:**
  - Visualize environment-specific metrics such as market state evolution, traffic density, and agent distances in robotics.

- **Results Export:**
  - Download simulation results as CSV files for further analysis.

## Installation

### Prerequisites

Ensure you have Python 3.x installed along with the following dependencies:


Additional Libraries:
streamlit
numpy
pandas
matplotlib
seaborn
torch
scikit-learn
networkx
plotly


streamlit run app.py

