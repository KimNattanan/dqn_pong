import pygame
from pygame.locals import *
import math
import random


WINDOW_W, WINDOW_H =  800, 480

class Player:
  def __init__(self,x,y,w,h):
    self.x, self.y = x,y
    self.w, self.h = w,h
    self.speed = 500  # px/s

  def move(self,dt,dx,dy):
    if(dx==0 and dy==0): return
    r = dt * self.speed / math.sqrt(dx**2+dy**2)
    self.x += dx*r
    self.y += dy*r
    self.x = max(0,min(self.x,WINDOW_W-self.w))
    self.y = max(0,min(self.y,WINDOW_H-self.h))

  def getRect(self):
    return [self.x,self.y,self.w,self.h]

class Ball:
  ping = 10
  def __init__(self,x,y,rad,dx,dy,plrs):
    self.x, self.y, self.rad = x,y,rad
    self.x0,self.y0 = x,y
    self.lz = False
    self.dx, self.dy = dx,dy
    self.plrs = plrs
    self.hits = [False]*len(plrs)
    self.speed = 1000

  def findEndPoint(self,x0,y0,x1,y1):
    if self.szLine(self.x0,self.y0,x1,y1) >= Ball.ping:
      x0,y0 = self.x0,self.y0
      self.lz = True
      if y1<0: # hit top
        ymid = 0
        xmid = (ymid-y1)/(y0-y1)*(x0-x1)+x1
        x2,y2,dx2,dy2 = self.findEndPoint(x0,y0,xmid,ymid)
        if x2!=xmid or y2!=ymid:
          sz = self.szLine(xmid,ymid,x1,y1)
          return self.findEndPoint(x2,y2,x2+dx2*sz,y2+dy2*sz)
        y1 = 2*ymid-y1
        return self.findEndPoint(xmid,ymid,x1,y1)
      if y1+2*self.rad-1 >= WINDOW_H: # hit bottom
        ymid = WINDOW_H-1-2*self.rad
        xmid = (y1-ymid)/(y1-y0)*(x0-x1)+x1
        x2,y2,dx2,dy2 = self.findEndPoint(x0,y0,xmid,ymid)
        if x2!=xmid or y2!=ymid:
          sz = self.szLine(xmid,ymid,x1,y1)
          return self.findEndPoint(x2,y2,x2+dx2*sz,y2+dy2*sz)
        y1 = 2*ymid-y1
        return self.findEndPoint(xmid,ymid,x1,y1)
      hit,x_hit,y_hit = self.checkHit(self.plrs[0],x0,y0,x1,y1)
      if hit: # hit left plr
        dx_hit = self.plrs[0].h/4
        dy_hit = min(max(y_hit - (self.plrs[0].y+self.plrs[0].h/2),-self.plrs[0].h/2),self.plrs[0].h/2)
        dx_hit,dy_hit = self.normVec(dx_hit,dy_hit)
        if self.hits[0]:
          return x1,y1,dx_hit,dy_hit
        self.hits[0] = True
        sz = self.szLine(x_hit,y_hit,x1,y1)
        sz = max(sz,1)
        return self.findEndPoint(x_hit,y_hit,x_hit+dx_hit*sz,y_hit+dy_hit*sz)
      hit,x_hit,y_hit = self.checkHit(self.plrs[1],x0,y0,x1,y1)
      if hit: # hit right plr
        dx_hit = -self.plrs[1].h/4
        dy_hit = min(max(y_hit - (self.plrs[1].y+self.plrs[1].h/2),-self.plrs[1].h/2),self.plrs[1].h/2)
        dx_hit,dy_hit = self.normVec(dx_hit,dy_hit)
        if self.hits[1]:
          return x1,y1,dx_hit,dy_hit
        self.hits[1] = True
        sz = self.szLine(x_hit,y_hit,x1,y1)
        sz = max(sz,1)
        return self.findEndPoint(x_hit,y_hit,x_hit+dx_hit*sz,y_hit+dy_hit*sz)
    dx,dy = self.normVec(x1-x0,y1-y0)
    return x1,y1,dx,dy
  
  def checkHit(self,plr,x0,y0,x1,y1):
    px0,py0,px1,py1 = plr.x,plr.y,plr.x+plr.w,plr.y+plr.h
    if self.rectXcircle(px0,py0,px1,py1,x0,y0,self.rad):
      return True,x0,y0
    if x0==x1 and y0==y1:
      return False,0,0
    
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx**2 + dy**2)
    px = -dy / length
    py = dx / length
    
    x0_0 = x0 + px * self.rad
    y0_0 = y0 + py * self.rad
    x1_0 = x1 + px * self.rad
    y1_0 = y1 + py * self.rad
    
    x0_1 = x0 - px * self.rad
    y0_1 = y0 - py * self.rad
    x1_1 = x1 - px * self.rad
    y1_1 = y1 - py * self.rad

    hits = self.rectXline(px0,py0,px1,py1,x0_0,y0_0,x1_0,y1_0)
    if len(hits):
      return True,*sorted(hits, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)[0]
    hits = self.rectXline(px0,py0,px1,py1,x0_1,y0_1,x1_1,y1_1)
    if len(hits):
      return True,*sorted(hits, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)[0]

    if self.rectXcircle(px0,py0,px1,py1,x1,y1,self.rad):
      return True,x1,y1

    return False,0,0
      



  def move(self,dt):
    if self.dx==0 and self.dy==0: return
    if dt==0: return
    sz = math.sqrt(self.dx**2+self.dy**2)
    r = dt * self.speed / sz
    self.x,self.y,self.dx,self.dy = self.findEndPoint(self.x,self.y,self.x+self.dx*r,self.y+self.dy*r)
    if self.lz:
      self.lz = False
      self.x0,self.y0 = self.x,self.y
    

  def rectXcircle(self,x0,y0,x1,y1,cx,cy,rad):
    # rect: (x0,y0,x1,y1), cirlce: (cx,cy,rad)
    closest_x = max(x0, min(cx+rad, x1))
    closest_y = max(y0, min(cy+rad, y1))
    distance_x = cx+rad - closest_x
    distance_y = cy+rad - closest_y
    return distance_x**2 + distance_y**2 <= rad**2
  def rectXline(self,x0,y0,x1,y1,x2,y2,x3,y3):
    # rect: (x0,y0,x1,y1), line: ((x2,y2),(x3,y3))
    if x0<=x2<=x1 and y0<=y2<=y1: return [(x2,y2),(x3,y3)]
    hits = [self.lineXline(x0,y0,x0,y1,x2,y2,x3,y3),
            self.lineXline(x0,y0,x1,y0,x2,y2,x3,y3),
            self.lineXline(x0,y1,x1,y1,x2,y2,x3,y3),
            self.lineXline(x1,y0,x1,y1,x2,y2,x3,y3)]
    return [e for e in hits if e]
  def lineXline(self,x0,y0,x1,y1,x2,y2,x3,y3):
    # line1: ((x0,y0),(x1,y1)), line2: ((x2,y2),(x3,y3))
    def det(a, b, c, d):
      return a*d - b*c

    a1 = y1 - y0
    b1 = x0 - x1
    c1 = a1*x0 + b1*y0

    a2 = y3-y2
    b2 = x2-x3
    c2 = a2*x2 + b2*y2

    determinant = det(a1,b1,a2,b2)
    if determinant == 0:
      return None
    x = det(c1,b1,c2,b2) / determinant
    y = det(a1,c1,a2,c2) / determinant

    def is_between(p, q, r):
      return min(p,q) <= r <= max(p,q)

    if is_between(x0, x1, x) and is_between(y0, y1, y) and is_between(x2, x3, x) and is_between(y2, y3, y):
      return (x, y)
    else:
      return None

  def getCenter(self):
    return (self.x+self.rad,self.y+self.rad)
  
  def normVec(self,x,y):
    if x==0 and y==0 : return (0, 0)
    len = math.sqrt(x**2+y**2)
    return (x/len, y/len)
  def getNormVelocity(self):
    return self.normVec(self.dx,self.dy)
  def szLine(self,x0,y0,x1,y1):
    return math.sqrt((x0-x1)**2+(y0-y1)**2)


class App:
  def __init__(self,display=True,fixed_dt=0):
    self.display = display
    self.dt = fixed_dt
    self.use_dt = self.dt==0
    self._running = True
    self._display_surf = None
    self.clock = pygame.time.Clock()
    self.size = self.w, self.h = WINDOW_W, WINDOW_H
    self.keyDown = {}
    self.up, self.down = [False,False],[False,False]
    
    self.plr1 = None
    self.plr2 = None
    self.ball = None
    self.reset()

  def reset(self):
    if self.plr1: del self.plr1
    if self.plr2: del self.plr2
    if self.ball: del self.ball

    self.plr1 = Player(50,WINDOW_H/2-50,3,100)
    self.plr2 = Player(WINDOW_W-51,WINDOW_H/2-50,3,100)
    self.up, self.down = [False,False],[False,False]
    ball_vx = random.randint(1,10)
    if random.randint(0,1) : ball_vx = -ball_vx
    ball_vy = random.randint(-5,5)
    ball_vx, ball_vy = 1,0
    self.ball = Ball(WINDOW_W/2-10,WINDOW_H/2-10,10,ball_vx,ball_vy,[self.plr1,self.plr2])

    self.clock.tick()

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
    if self.display and self.use_dt:
      self.dt = self.clock.tick()/1000
    dy1,dy2 = 0,0
    if self.getKeyDown(pygame.K_w) or self.up[0]:
      dy1 += -1
    if self.getKeyDown(pygame.K_s) or self.down[0]:
      dy1 += 1
    if self.getKeyDown(pygame.K_UP) or self.up[1]:
      dy2 += -1
    if self.getKeyDown(pygame.K_DOWN) or self.down[1]:
      dy2 += 1
    # if self.ball.y+self.ball.rad < self.plr2.y+self.plr2.h/2:  # bot
    #   dy2 += -1
    # else:
    #   dy2 += 1
    self.plr1.move(self.dt,0,dy1)
    self.plr2.move(self.dt,0,dy2)
    self.ball.move(self.dt)
    
  def on_render(self):
    if not self.display: return
    self._display_surf.fill((0,0,0))
    pygame.draw.rect(self._display_surf,(255,255,255),self.plr1.getRect())
    pygame.draw.rect(self._display_surf,(255,255,255),self.plr2.getRect())
    pygame.draw.circle(self._display_surf,(255,0,0),self.ball.getCenter(),self.ball.rad)
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


class Game:
  STATE_SHAPE = 6
  ACTION_SHAPE = 3
  def __init__(self,display=True,fixed_dt=0):
    self.app = App(display,fixed_dt)
    self.force_quit = False

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
    self.app.reset()

  def cleanup(self):
    self.app.on_cleanup()

  def getState(self):
    return (*self.app.ball.getCenter(), *self.app.ball.getNormVelocity(), self.app.plr1.y+self.app.plr1.h/2, self.app.plr2.y+self.app.plr2.h/2)
  def isGameOver(self):
    if self.app.ball.x+self.app.ball.rad<0 or self.app.ball.x+self.app.ball.rad>WINDOW_W:
      return 1
    return 0
  def getReward(self,id):
    if self.app.ball.x+self.app.ball.rad<0:
      if id==0: return -10
      return 10
    if self.app.ball.x+self.app.ball.rad>WINDOW_W:
      if id==0: return 10
      return -10
    if self.app.ball.hits[id]:
      self.app.ball.hits[id] = False
      return 2
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
