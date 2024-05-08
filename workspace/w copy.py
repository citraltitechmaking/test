import sys
sys.setrecursionlimit(10 ** 5 + 10000)
# sys.setrecursionlimit(10**6)
input = sys.stdin.readline
INF = float('inf')
from heapq import heapify, heappop, heappush
import decimal
from decimal import Decimal
import math
from math import ceil, floor      # -(-a//b)で切り上げ、a//bで切り捨て      # SortedSetの内部で使用
from math import log2, log, sqrt
from math import gcd
def lcm(x, y): return (x * y) // gcd(x, y)
from itertools import combinations as comb                      # 重複なし組み合わせ    comb(L,2)
from itertools import combinations_with_replacement as comb_w   # 重複あり組み合わせ
from itertools import accumulate                                # 累積和
from itertools import product                                   # 直積  ans=list(itertools.product(L,M))
from itertools import permutations                              # 順列
from collections import deque, defaultdict, Counter
# import operator               # 削除予定
from copy import deepcopy
from bisect import bisect_left, bisect_right, insort
from typing import Generic, Iterable, Iterator, TypeVar, Union, List      # SortedSetの内部で使用
T = TypeVar('T')                                                          # SortedSetの内部で使用

MOD = 10**9+7
MOD2 = 998244353
alphabet = "abcdefghijklmnopqrstuvwxyz"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEBUG = True
 

def solve():

    h,w = MI()
    S = LLS(h)

    A = []
    for l in S:
        if "#" in l:
            A.append(l)
        else:
            continue

    B = list(zip(*A))

    C = []
    for l in B:
        if "#" in l:
            C.append(l)
        else:
            continue

    D = list(zip(*C))

    for d in D:
        print(''.join(d))
    
    



DEBUG = False

###########################デバッグ：色がつく ###########################
def debug(*args):
    if DEBUG:
        print('\033[92m', end='')
        print(*args)
        print('\033[0m', end='')


########################### 1行ショトカ ###########################
def int1(x): return int(x) - 1
def II(): return int(input())
def MI(): return map(int, input().split())
def MI1(): return map(int1, input().split())
def LI(): return list(map(int, input().split()))
def LI1(): return list(map(int1, input().split()))
def LIS(): return list(map(int, SI()))
def LA(f): return list(map(f, input().split()))
def LLI(H): return [LI() for _ in range(H)]     # H:列数
def SI(): return input().strip('\n')
def MS(): return input().split()
def LS(): return list(input().strip('\n'))
def LLS(H): return [LS() for _ in range(H)]
def gen_matrix(h, w, init): return [[init] * w for _ in range(h)]
def yes(): print('Yes'); exit()
def no(): print('No'); exit()

########################### 以下memo ###########################

### スニペット
#   UnionFind
#   dfs-graph: グラフ
#   dfs-maze : 迷路
#   input_graph: グラフ入力
#   input_tree : 木入力
#   list2: 二次元リスト入力
#   list2_sort: 二次元リストソート
#   meguru-2butannsanku: めぐる式二部探索
#   meguru-3butannsanku: めぐる式三部探索
#   BITS-pattern: bit全探索
#   SortedSet: 順序付き集合
#   SortedMultiSet: 順序付き多重集合
#   segtree: セグメント木
#   lazy_segtree: 遅延セグメント木

        
### memo
# At = list(zip(*A)) 転置行列
# nCr ans=math.comb(n,r)        # これはだめ。高速nCrを使用すること
        
### 迷路の前後左右
#for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
# for i,j in [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]:
        
### 数字文字交じりクエリを文字列のリストにする '1 19 G' -> ['1', '19', 'G']
# input()を含まず、受け取ったLLSのクエリの文字列に対し実行する
# l = ''.join(Strings).split(' ')
        
### UnionFind
# スニペット（unionfind）に使用例
# https://qiita.com/uniTM/items/77ef2412e426cae44586

### 重み付きUnionFind
# 出題数が少ないので優先度低
# 例：AさんはBさんより3歳上、BさんはCさんより2歳下。ではAさんはCさんよりいくつ差があるか？などが解ける 
# https://at274.hatenablog.com/entry/2018/02/03/140504



if __name__ == '__main__':
    solve()