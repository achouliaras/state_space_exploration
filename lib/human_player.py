
import pygame

pygame.init()

class Agent:
    def __init__(self, name, action_space, cfg):
        """Initialize a Player controlled agent
        """
        self.name = name
        self.action_space = action_space
        print(action_space)
        self.cfg = cfg
        self.action_queue = []

    def get_action(self):
        if self.cfg.domain == 'ALE':
            return self.get_action_atari()
        elif self.cfg.domain == 'MiniGrid':
            return self.get_action_minigrid()
        
    def get_action_atari(self):
        """
        Returns the action that the user chooses
        """
        action = 0
        
        # while(True):
        pygame.event.clear()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE]:
            action = 1
        elif pressed[pygame.K_w]:
            action = 5
        elif pressed[pygame.K_a]:
            action = 3
        elif pressed[pygame.K_s]:
            action = 4
        elif pressed[pygame.K_d]:
            action = 2
        if self.action_space.contains(action): return action
        print('Wrong Action')
        return 0

    def get_action_minigrid(self):
        """
        Returns the action that the user chooses
        """
        if len(self.action_queue) != 0:
            return self.action_queue.pop(0)

        pygame.event.clear()
        while(True):
            if len(self.action_queue) != 0:
                return self.action_queue.pop(0)
        
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and len(self.action_queue) != 0:
                    return self.action_queue.pop(0)
                
                if event.key == pygame.K_SPACE:
                    self.action_queue.append(3) # Pickup item
                elif event.key == pygame.K_w:
                    self.action_queue.append(2) 
                elif event.key == pygame.K_a:
                    self.action_queue.append(0)
                elif event.key == pygame.K_d:
                    self.action_queue.append(1) 
                elif event.key == pygame.K_s:
                    self.action_queue.append(4) # Drop item
                elif event.key == pygame.K_e:
                    self.action_queue.append(5) # Open door
                elif event.key == pygame.K_f:
                    self.action_queue.append(6)
          
    def get_name(self):
        return self.name

