import pygame
from pygame.locals import *
import math
import numpy as np
import random


WINDOW_W, WINDOW_H =  800, 480

class Player:
  def __init__(self,x,y,w,h):
    self.x, self.y = x,y
    self.w, self.h = w,h
    self.speed = 2

  def move(self,dx,dy):
    self.x = max(0,min(self.x+dx*self.speed,WINDOW_W-self.w))
    self.y = max(0,min(self.y+dy*self.speed,WINDOW_H-self.h))

  def getRect(self):
    return (self.x,self.y,self.w,self.h)
  
  def isInside(self,x,y):
    return 0<=x-self.x<=self.w and 0<=y-self.y<=self.h

class Ball:
  def __init__(self,x,y,rad,dx,dy):
    self.x, self.y, self.rad = x,y,rad
    self.speed = 4
    self.dx, self.dy = dx,dy
  
  def move(self):
    self.x += self.dx*self.speed
    self.y += self.dy*self.speed
    if self.y-self.rad<0:
      self.y = 2*self.rad-self.y
      self.dy = -self.dy
    if self.y+self.rad>WINDOW_H:
      self.y = 2*(WINDOW_H-self.rad)-self.y
      self.dy = -self.dy
      


class App:
  def __init__(self,display=True):
    self.display = display
    self._running = True
    self._display_surf = None
    self.size = self.w, self.h = WINDOW_W, WINDOW_H
    self.keyDown = {}
    
    pygame.font.init()
    self.plr1 = None
    self.plr2 = None
    self.ball = None
    self.up, self.down, self.hits = None,None,None
    self.score1, self.score2 = None,None
    self.reset(np.array([0,0]))

  def reset(self,scores):
    if self.plr1: del self.plr1
    if self.plr2: del self.plr2
    if self.ball: del self.ball

    self.plr1 = Player(50,WINDOW_H/2-50,10,100)
    self.plr2 = Player(WINDOW_W-50,WINDOW_H/2-50,10,100)
    self.up, self.down, self.hits = np.array([False]*2),np.array([False]*2),np.array([0]*2)
    ball_vx = random.randint(1,10)
    if random.randint(0,1) : ball_vx = -ball_vx
    ball_vy = random.randint(-5,5)
    ball_vx,ball_vy = self.normVec((ball_vx,ball_vy))
    self.ball = Ball(WINDOW_W/2,WINDOW_H/2,10,ball_vx,ball_vy)

    score_font = pygame.font.SysFont('Comic Sans MS', 30)
    self.score1 = score_font.render(str(scores[0]),False,(0,255,0))
    self.score2 = score_font.render(str(scores[1]),False,(0,255,0))

  def on_init(self):
    pygame.init()
    if self.display:
      self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    self._running = True

  def on_event(self, event):
    if not self.display: return
    if event.type == pygame.QUIT:
      self._running = False
    elif event.type == pygame.KEYDOWN:
      self.keyDown[event.key] = True
    elif event.type == pygame.KEYUP:
      self.keyDown[event.key] = False
  
  def getKeyDown(self,key):
    if not self.display: return False
    return self.keyDown.get(key,False)

  def on_loop(self):
    dy1,dy2 = 0,0
    if self.getKeyDown(pygame.K_w) or self.up[0]:
      dy1 += -1
    if self.getKeyDown(pygame.K_UP) or self.up[1]:
      dy2 += -1
    if self.getKeyDown(pygame.K_s) or self.down[0]:
      dy1 += 1
    if self.getKeyDown(pygame.K_DOWN) or self.down[1]:
      dy2 += 1
    self.plr1.move(0,dy1)
    self.plr2.move(0,dy2)
    self.ball.move()

    if self.rectXcircle(self.plr1.x,self.plr1.y,self.plr1.x+self.plr1.w,self.plr1.y+self.plr1.h,self.ball.x,self.ball.y,self.ball.rad):
      dx = self.plr1.h/4
      dy = self.ball.y - (self.plr1.y+self.plr1.h/2)
      dx,dy = self.normVec((dx,dy))
      self.ball.dx, self.ball.dy = dx,dy
      if self.hits[0]==0:
        self.hits[0] = 1
    elif self.hits[0]==2:
      self.hits[0] = 0
    if self.rectXcircle(self.plr2.x,self.plr2.y,self.plr2.x+self.plr2.w,self.plr2.y+self.plr2.h,self.ball.x,self.ball.y,self.ball.rad):
      dx = -self.plr2.h/4
      dy = self.ball.y - (self.plr2.y+self.plr2.h/2)
      dx,dy = self.normVec((dx,dy))
      self.ball.dx, self.ball.dy = dx,dy
      if self.hits[1]==0:
        self.hits[1] = 1
    elif self.hits[1]==2:
      self.hits[1] = 0
    
  def on_render(self):
    if not self.display: return
    self._display_surf.fill((0,0,0))
    self._display_surf.blit(self.score1,(WINDOW_W*0.25-self.score1.get_width()/2,0))
    self._display_surf.blit(self.score2,(WINDOW_W*0.75-self.score2.get_width()/2,0))
    pygame.draw.rect(self._display_surf,(255,255,255),self.plr1.getRect()) #plr1
    pygame.draw.rect(self._display_surf,(255,255,255),self.plr2.getRect()) #plr2
    pygame.draw.circle(self._display_surf,(255,0,0),(self.ball.x,self.ball.y),self.ball.rad) # ball
    pygame.display.flip()
    
  def on_cleanup(self):
    pygame.quit()

  def on_execute(self):
    if self.on_init() == False:
      self._running = False

    while( self._running ):
      if self.display:
        for event in pygame.event.get():
          self.on_event(event)
      self.on_loop()
      self.on_render()
    self.on_cleanup()

  def normVec(self,vec):
    sz = math.sqrt(vec[0]**2+vec[1]**2)
    return (vec[0]/sz, vec[1]/sz)
  
  def rectXcircle(self,x0,y0,x1,y1,cx,cy,rad):
    # rect: (x0,y0,x1,y1), cirlce: (cx,cy,rad)
    closest_x = max(x0, min(cx, x1))
    closest_y = max(y0, min(cy, y1))
    distance_x = cx - closest_x
    distance_y = cy - closest_y
    return distance_x**2 + distance_y**2 <= rad**2


class Game:
  STATE_SHAPE = 6
  ACTION_SHAPE = 3
  def __init__(self,display=True):
    self.app = App(display)
    self.force_quit = False
    self.scores = np.array([0,0])

    self.app.on_init()

  
  def upd(self):
    if self.app.display:
      for event in pygame.event.get():
        self.app.on_event(event)
      if self.app.getKeyDown(pygame.K_ESCAPE):
        self.force_quit = True
    self.app.on_loop()
    self.app.on_render()

  def reset(self):
    self.app.reset(self.scores)

  def cleanup(self):
    self.app.on_cleanup()

  def getState(self):
    return (self.app.ball.x,self.app.ball.y, self.app.ball.dx,self.app.ball.dy, self.app.plr1.y+self.app.plr1.h/2, self.app.plr2.y+self.app.plr2.h/2)
  def isGameOver(self):
    if self.app.ball.x<0 or self.app.ball.x>WINDOW_W:
      return 1
    return 0
  def getReward(self,id):
    if self.app.ball.x<0:
      if id==0: return -3
      return 3
    if self.app.ball.x>WINDOW_W:
      if id==0: return 3
      return -3
    if self.app.hits[id]:
      self.app.hits[id] = 2
      return 1
    return 0
  
  def up(self,id):
    self.app.up[id] = True
    self.app.down[id] = False
  def idle(self,id):
    self.app.up[id] = False
    self.app.down[id] = False
  def down(self,id):
    self.app.up[id] = False
    self.app.down[id] = True
