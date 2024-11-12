
import pygame

pygame.init()

class Agent:
    def __init__(self, name, action_space):
        """Initialize a Player controlled agent
        """
        self.name = name
        self.action_space = action_space
        print(action_space)
        self.action_queue = []

    def get_action(self):
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
            
        pygame.event.clear()
        while(True):
            if len(self.action_queue) != 0:
                return self.action_queue.pop(0)

            event = pygame.event.poll()
            if event == pygame.NOEVENT: return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and len(self.action_queue) != 0:
                    return self.action_queue.pop(0)
                
                if event.key == pygame.K_SPACE:
                    # self.action_queue.append(1)
                    return 1
                elif event.key == pygame.K_w:
                    # self.action_queue.append(2)
                    return 2
                elif event.key == pygame.K_a:
                    # self.action_queue.append(3)
                    return 3
                elif event.key == pygame.K_s:
                    # self.action_queue.append(5)
                    return 5
                elif event.key == pygame.K_d:
                    # self.action_queue.append(2)
                    return 2
                else:
                    # self.action_queue.append(0)
                    return 0
            # else:
            #     self.action_queue.append(0)

    # def get_action(self):
    #     """
    #     Returns the action that the user chooses
    #     """
    #     if len(self.action_queue) != 0:
    #         return self.action_queue.pop(0)

    #     pygame.event.clear()
    #     while(True):
    #         if len(self.action_queue) != 0:
    #             return self.action_queue.pop(0)
        
    #         event = pygame.event.wait()
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_RETURN and len(self.action_queue) != 0:
    #                 return self.action_queue.pop(0)
                
    #             if event.key == pygame.K_SPACE:
    #                 self.action_queue.append(1)
    #             elif event.key == pygame.K_w:
    #                 self.action_queue.append(2)
    #             elif event.key == pygame.K_a:
    #                 self.action_queue.append(3)
    #             elif event.key == pygame.K_s:
    #                 self.action_queue.append(5)
    #             elif event.key == pygame.K_d:
    #                 self.action_queue.append(2)
    #             else:
    #                 self.action_queue.append(0)
            
    def get_name(self):
        return self.name

