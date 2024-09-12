import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import requests
import json
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any

# Google Gemini API configuration
API_KEY = "enter your gemini api here"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Environment classes
class BaseEnvironment(ABC):
    def __init__(self, num_agents: int, max_steps: int):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.info_history: List[Dict[str, Any]] = []

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, actions):
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        pass

class EconomicEnvironment(BaseEnvironment):
    def __init__(self, num_agents: int = 5, max_steps: int = 100):
        super().__init__(num_agents, max_steps)
        self._observation_space = 5
        self._action_space = 5
        self.market_state = np.random.rand(5)
    
    def reset(self):
        self.current_step = 0
        self.market_state = np.random.rand(5)
        self.info_history = []
        return [np.random.rand(self._observation_space) for _ in range(self.num_agents)]
    
    def step(self, actions):
        self.current_step += 1
        self.market_state += np.mean(actions, axis=0) * 0.1
        self.market_state = np.clip(self.market_state, 0, 1)
        observations = [np.concatenate([self.market_state, np.random.rand(self._observation_space - 5)]) for _ in range(self.num_agents)]
        rewards = [np.dot(action, self.market_state) for action in actions]
        done = self.current_step >= self.max_steps
        info = {"market_state": self.market_state.copy()}
        self.info_history.append(info)
        return observations, rewards, done, info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

class TrafficEnvironment(BaseEnvironment):
    def __init__(self, num_agents: int = 5, max_steps: int = 100):
        super().__init__(num_agents, max_steps)
        self._observation_space = 4
        self._action_space = 4
        self.traffic_density = np.random.rand(4)
    
    def reset(self):
        self.current_step = 0
        self.traffic_density = np.random.rand(4)
        self.info_history = []
        return [np.concatenate([self.traffic_density, np.random.rand(self._observation_space - 4)]) for _ in range(self.num_agents)]
    
    def step(self, actions):
        self.current_step += 1
        self.traffic_density += np.mean(actions, axis=0) * 0.1
        self.traffic_density = np.clip(self.traffic_density, 0, 1)
        observations = [np.concatenate([self.traffic_density, np.random.rand(self._observation_space - 4)]) for _ in range(self.num_agents)]
        rewards = [1 - np.linalg.norm(action - self.traffic_density) for action in actions]
        done = self.current_step >= self.max_steps
        info = {"traffic_density": self.traffic_density.copy()}
        self.info_history.append(info)
        return observations, rewards, done, info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

class RoboticsEnvironment(BaseEnvironment):
    def __init__(self, num_agents: int = 5, max_steps: int = 100):
        super().__init__(num_agents, max_steps)
        self._observation_space = 6
        self._action_space = 6
        self.target_position = np.random.rand(3)
    
    def reset(self):
        self.current_step = 0
        self.target_position = np.random.rand(3)
        self.info_history = []
        return [np.concatenate([self.target_position, np.random.rand(self._observation_space - 3)]) for _ in range(self.num_agents)]
    
    def step(self, actions):
        self.current_step += 1
        observations = [np.concatenate([self.target_position, np.random.rand(self._observation_space - 3)]) for _ in range(self.num_agents)]
        distances = [np.linalg.norm(action[:3] - self.target_position) for action in actions]
        rewards = [1 / (1 + distance) for distance in distances]
        done = self.current_step >= self.max_steps
        info = {"target_position": self.target_position.copy(), "distances": distances}
        self.info_history.append(info)
        return observations, rewards, done, info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

# Agent classes
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, observation_space: int, action_space: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(observation_space, action_space).to(self.device)
        self.target_dqn = DQN(observation_space, action_space).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_space)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn(state)
        return q_values.cpu().data.numpy()[0]
    
    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.dqn(states)
        next_q_values = self.target_dqn(next_states).detach()
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1).expand_as(current_q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        if np.random.rand() < 0.001:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

class SupervisedAgent:
    def __init__(self, observation_space: int, action_space: int):
        self.model = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
    
    def act(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            return self.model(state).numpy()
    
    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        self.optimizer.zero_grad()
        predicted_action = self.model(state)
        loss = self.criterion(predicted_action, action)
        loss.backward()
        self.optimizer.step()

# Visualization and Analysis
class Visualizer:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.agent_actions = []
    
    def update(self, episode: int, reward: float, actions: np.ndarray):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.agent_actions.append(actions)
    
    def plot_rewards(self):
        fig = px.line(x=self.episodes, y=self.rewards, labels={'x': 'Episode', 'y': 'Total Reward'})
        fig.update_layout(title='Total Reward per Episode')
        return fig

    def plot_action_heatmap(self):
        actions = np.array(self.agent_actions)
        fig = px.imshow(actions.mean(axis=0), labels=dict(x="Action Dimension", y="Agent", color="Average Action"))
        fig.update_layout(title='Average Agent Actions Heatmap')
        return fig

    def plot_action_tsne(self):
        if not self.agent_actions:
            return go.Figure()
        actions = np.array(self.agent_actions).reshape(-1, np.array(self.agent_actions).shape[-1])
        tsne = TSNE(n_components=2, random_state=42)
        actions_2d = tsne.fit_transform(actions)
        fig = px.scatter(x=actions_2d[:, 0], y=actions_2d[:, 1], labels={'x': 't-SNE 1', 'y': 't-SNE 2'})
        fig.update_layout(title='t-SNE Visualization of Agent Actions')
        return fig

    def plot_agent_clusters(self):
        if not self.agent_actions:
            return go.Figure()
        actions = np.array(self.agent_actions).reshape(-1, np.array(self.agent_actions).shape[-1])
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(actions)
        fig = px.scatter(x=actions[:, 0], y=actions[:, 1], color=clusters, labels={'x': 'Action Dimension 1', 'y': 'Action Dimension 2', 'color': 'Cluster'})
        fig.update_layout(title='Agent Clusters based on Actions')
        return fig

    def plot_reward_distribution(self):
        fig = px.histogram(x=self.rewards, nbins=30, labels={'x': 'Reward', 'y': 'Frequency'})
        fig.update_layout(title='Reward Distribution')
        return fig

# Network Analysis class
class NetworkAnalysis:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.G = nx.Graph()
        self.G.add_nodes_from(range(num_agents))

    def update(self, actions: np.ndarray):
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                similarity = 1 / (1 + np.linalg.norm(actions[i] - actions[j]))
                self.G.add_edge(i, j, weight=similarity)

    def plot_network(self):
        pos = nx.spring_layout(self.G)
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[node][0] for node in self.G.nodes()]
        node_y = [pos[node][1] for node in self.G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        node_adjacencies = []
        node_texts = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_texts.append(f'Agent {node}: # of connections: {len(adjacencies[1])}')
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_texts
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Agent Interaction Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

# Google Gemini API
def query_gemini(query: str, env_info: str, agent_info: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{
                "text": f"Environment: {env_info}\nAgents: {agent_info}\nQuery: {query}\nProvide insights and explanations about the agent strategies."
            }]
        }]
    }
    
    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "Error querying Gemini API"

# Main application
def main():
    st.title("Advanced Multi-Agent Simulation Tool")

    # Sidebar for configuration
    st.sidebar.header("Simulation Configuration")
    env_type = st.sidebar.selectbox("Select Environment", ["Economic", "Traffic", "Robotics"])
    num_agents = st.sidebar.slider("Number of Agents", 1, 20, 5)
    agent_type = st.sidebar.selectbox("Agent Type", ["RL", "Supervised"])
    num_episodes = st.sidebar.slider("Number of Episodes", 100, 10000, 1000)
    max_steps = st.sidebar.slider("Max Steps per Episode", 50, 500, 100)

    # Create environment
    if env_type == "Economic":
        env = EconomicEnvironment(num_agents, max_steps)
    elif env_type == "Traffic":
        env = TrafficEnvironment(num_agents, max_steps)
    else:
        env = RoboticsEnvironment(num_agents, max_steps)

    # Create agents
    if agent_type == "RL":
        agents = [RLAgent(env.observation_space, env.action_space) for _ in range(num_agents)]
    else:
        agents = [SupervisedAgent(env.observation_space, env.action_space) for _ in range(num_agents)]

    # Run simulation
    if st.sidebar.button("Start Simulation"):
        visualizer = Visualizer()
        network_analysis = NetworkAnalysis(num_agents)
        
        progress_bar = st.progress(0)
        episode_reward_placeholder = st.empty()
        
        for episode in range(num_episodes):
            observations = env.reset()
            done = False
            total_reward = 0
            episode_actions = []
            
            while not done:
                actions = [agent.act(obs) for agent, obs in zip(agents, observations)]
                next_observations, rewards, done, info = env.step(actions)
                
                for agent, obs, action, reward, next_obs in zip(agents, observations, actions, rewards, next_observations):
                    agent.learn(obs, action, reward, next_obs, done)
                
                observations = next_observations
                total_reward += sum(rewards)
                episode_actions.append(actions)
            
            mean_actions = np.mean(episode_actions, axis=0)
            visualizer.update(episode, total_reward, mean_actions)
            network_analysis.update(mean_actions)
            
            progress = (episode + 1) / num_episodes
            progress_bar.progress(progress)
            episode_reward_placeholder.text(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")
        
        st.success("Simulation completed!")

        # Visualizations
        st.subheader("Performance Analysis")
        st.plotly_chart(visualizer.plot_rewards())
        st.plotly_chart(visualizer.plot_action_heatmap())

        st.subheader("Agent Behavior Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(visualizer.plot_action_tsne())
        with col2:
            st.plotly_chart(visualizer.plot_agent_clusters())

        st.subheader("Reward Distribution")
        st.plotly_chart(visualizer.plot_reward_distribution())

        st.subheader("Agent Interaction Network")
        st.plotly_chart(network_analysis.plot_network())

        # Environment-specific analysis
        st.subheader("Environment-Specific Analysis")
        if env_type == "Economic":
            market_states = [info["market_state"] for info in env.info_history]
            market_df = pd.DataFrame(market_states, columns=[f"Asset {i+1}" for i in range(5)])
            st.plotly_chart(px.line(market_df, labels={'index': 'Step', 'value': 'Asset Value', 'variable': 'Asset'}))
            st.write("Market State Evolution")
        elif env_type == "Traffic":
            traffic_densities = [info["traffic_density"] for info in env.info_history]
            traffic_df = pd.DataFrame(traffic_densities, columns=[f"Road {i+1}" for i in range(4)])
            st.plotly_chart(px.line(traffic_df, labels={'index': 'Step', 'value': 'Traffic Density', 'variable': 'Road'}))
            st.write("Traffic Density Evolution")
        else:  # Robotics
            target_positions = [info["target_position"] for info in env.info_history]
            distances = [info["distances"] for info in env.info_history]
            pos_df = pd.DataFrame(target_positions, columns=["X", "Y", "Z"])
            dist_df = pd.DataFrame(distances).T
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Target Position Evolution", "Agent Distances to Target"))
            fig.add_trace(px.line(pos_df).data[0], row=1, col=1)
            fig.add_trace(px.line(dist_df).data[0], row=2, col=1)
            st.plotly_chart(fig)

        # Query Gemini for strategy explanation
        st.subheader("AI-Generated Strategy Insights")
        query = "What strategies did agents use to optimize the outcome?"
        response = query_gemini(query, str(env), str(agents))
        st.write(response)
        
        # Additional analysis
        st.subheader("Performance Metrics")
        df = pd.DataFrame({"Episode": visualizer.episodes, "Total Reward": visualizer.rewards})
        st.plotly_chart(px.line(df, x="Episode", y="Total Reward"))
        
        st.subheader("Agent Comparison")
        agent_rewards = [sum(visualizer.rewards[i::num_agents]) for i in range(num_agents)]
        st.plotly_chart(px.bar(x=range(num_agents), y=agent_rewards, labels={'x': 'Agent', 'y': 'Total Reward'}))
        
        st.subheader("Environment Statistics")
        st.write(f"Observation Space: {env.observation_space}")
        st.write(f"Action Space: {env.action_space}")
        st.write(f"Number of Steps: {env.max_steps}")

        # Download results
        results_df = pd.DataFrame({
            "Episode": visualizer.episodes,
            "Total Reward": visualizer.rewards,
            "Agent Actions": visualizer.agent_actions
        })
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Simulation Results",
            data=csv,
            file_name="simulation_results.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
