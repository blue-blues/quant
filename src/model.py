import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging

class DQN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 256, 128]):
        super(DQN, self).__init__()
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),  # Replace BatchNorm with LayerNorm
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.LayerNorm(hidden_sizes[i+1]),  # Replace BatchNorm with LayerNorm
                nn.ReLU(),
                nn.Dropout(0.2)
            ))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 3)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class Agent:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # Create models and move to device
        self.model = DQN(input_size=config.INPUT_SIZE).to(self.device)
        self.target_model = DQN(input_size=config.INPUT_SIZE).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
        
        self.current_lr = config.LEARNING_RATE
        
        self.steps = 0
        self.update_target_steps = config.TARGET_UPDATE_STEPS
        self.logger = logging.getLogger(__name__)

    def get_current_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.current_lr
    
    def step_scheduler(self, metric):
        """Step the scheduler and update current learning rate"""
        self.scheduler.step(metric)
        self.current_lr = self.optimizer.param_groups[0]['lr']
        return self.current_lr

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 2)
        
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.cpu().argmax().item()
    
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        self.model.train()  # Set to training mode
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Move tensors to GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + (1 - dones.float()) * self.config.GAMMA * next_q_values.squeeze()
        
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model with dimension compatibility check"""
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(path, weights_only=True)
            
            # Check model state dict
            if 'model_state_dict' in checkpoint:
                # Get the input size from the checkpoint
                input_layer_weight = checkpoint['model_state_dict']['input_layer.0.weight']
                checkpoint_input_size = input_layer_weight.shape[1]
                
                if checkpoint_input_size != self.config.INPUT_SIZE:
                    self.logger.warning(
                        f"Model input size mismatch: checkpoint has {checkpoint_input_size} features, "
                        f"but environment expects {self.config.INPUT_SIZE}. Creating new model."
                    )
                    # Initialize a new model with correct dimensions
                    self.model = DQN(input_size=self.config.INPUT_SIZE)
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
                else:
                    # Load the checkpoint if dimensions match
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            else:
                self.logger.warning("Invalid checkpoint format. Creating new model.")
                self.model = DQN(input_size=self.config.INPUT_SIZE)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
                
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {str(e)}")
            self.logger.info("Initializing new model with current configuration.")
            self.model = DQN(input_size=self.config.INPUT_SIZE)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Always ensure the model is in eval mode after loading
        self.model.eval()
        return self
