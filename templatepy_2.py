import sys
sys.setrecursionlimit(10 ** 5 + 10000)
# sys.setrecursionlimit(10**6)
input = sys.stdin.readline
INF = float('inf')
from heapq import heapify, heappop, heappush
import decimal
from decimal import Decimal
import math
# from math import ceil, floor      # -(-a//b)で切り上げ、a//bで切り捨て
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
from typing import Generic, Iterable, Iterator, TypeVar, Union, List
T = TypeVar('T')

MOD = 10**9+7
MOD2 = 998244353
alphabet = "abcdefghijklmnopqrstuvwxyz"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEBUG = True
 

def solve():

    n = II()
    n,m = MI()



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

########################### 以下テンプレート ###########################
        
### memo
# At = list(zip(*A)) 転置行列
# nCr ans=math.comb(n,r)        # これはだめ。高速nCrを使用すること
        
### 迷路の前後左右
#for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        
### 数字文字交じりクエリを文字列のリストにする '1 19 G' -> ['1', '19', 'G']
# input()を含まず、受け取ったLLSのクエリの文字列に対し実行する
# l = ''.join(Strings).split(' ')
        
### スニペット
#   関数化が難しいものを一連のコードとして使用する
#   BITS-pattern: bit全探索

### テンプレート
#       一連の入出力が明確なものは関数やクラスで呼び出すだけにしたい
#       そういったものはテンプレート

### UnionFind
# スニペット（unionfind）に使用例
# https://qiita.com/uniTM/items/77ef2412e426cae44586
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    # xの属する根を求める
    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    # findと同じだけど、rootという名前でも用意しておく（追記）
    def root(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    # 併合
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    # xとyが同じ集合に属するかを判定（同じ根に属するか）
    def same(self, x, y):
        return self.find(x) == self.find(y)

    # xが属するグループの全メンバを列挙
    # O(N)なので注意！
    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    # UnionFindに含まれるすべての根を列挙
    # O(N)なので注意！
    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    # 同じくO(N)なので注意！
    def group_count(self):
        return len(self.roots())

    # 同じくO(N)なので注意！
    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())

### 重み付きUnionFind
# 出題数が少ないので優先度低
# 例：AさんはBさんより3歳上、BさんはCさんより2歳下。ではAさんはCさんよりいくつ差があるか？などが解ける 
# https://at274.hatenablog.com/entry/2018/02/03/140504


### ダイクストラ法
#   重み付きグラフの最短経路を全頂点に対し得られる
# ----------------------------------------------------------------
# Input
#   1. タプル(行先, 重み)の二次元配列(隣接リスト)
#   2. 探索開始ノード(番号)
# Output
#   スタートから各ノードへの最小コストのリスト
# 注意点
#   コストが負の時は使えない
# ----------------------------------------------------------------
def dijkstra(edges: "List[List[(to, cost)]]", start_node: int) -> list:
    ### ポイント１：ダイクストラではheapqで管理する
    hq = []
    heapify(hq)

    ### ポイント２：各頂点間の距離をdistで管理。BFSと同じ
    dist = [INF] * len(edges)
    heappush(hq, (0, start_node))       # (距離、頂点)にすると距離でheapしてくれる
    dist[start_node] = 0

    ### dijkstra
    while hq:
        dist_from, v_from = heappop(hq)
        ### 処理するか判断
        # hqに同じ頂点がたくさん入ってる場合がある
        if dist_from > dist[v_from]:
            # 最もよい辺を発見後は dist_from == dist[v_from]
            # そうでないときは処理をスキップ
            continue
        ### 頂点訪問＆次の候補探し
        for v_next, cost_to_next in edges[v_from]:
        ### コスト計算にひと手間いる場合 (192 E)
        # for v_next, cost_info in edges[v_from]:
        #     cost, interval = cost_info
            if dist[v_next] > dist[v_from] + cost_to_next:
                dist[v_next] = dist[v_from] + cost_to_next
                heappush(hq, (dist[v_next], v_next))
    return dist

def memo_use_dijkstra():
    # 以下　ダイクストラ使用例
    n, m = MI()

    edge = [[] for _ in range(n)]
    # ダイクストラ法のためのリストを作成
    # i頂点目から伸びる頂点を (行先, コスト) の形式で追加
    for _ in range(m):
        a, b, t = MI()
        a, b = a-1, b-1
        edge[a].append((b, t))
        edge[b].append((a, t))
        
        ### コスト計算にひと手間必要な場合はコストをタプルにして、受け取り側でコスト算出
        # cost = (t, k)
        # edge[a].append((b, cost))
    # ここでダイクストラ実行
    dist = dijkstra(edge, 0)

    for x in dist:
        print(x if x != INF else -1)

    ### ダイクストラメモ
    # ABC252 E: 重み付き最短経路はダイクストラ
    #           クラスカル法の最小全域木は近似値でしかない（１敗）
    #           ひとひねり。returnをダイクストラ後のグラフと改変
        

### 素数関連
# 素因数分解 -> dict
def prime_factorize(n):
    #assert 1 <= n
    dic = dict()
    for p in range(2, n+1):
        if p * p > n:
            break
        if n % p == 0:
            cnt = 0
            while n % p == 0:
                cnt += 1
                n //= p
            dic[p] = cnt
    if n > 1:
        dic[n] = 1
    return dic

    prime_factorization(12)     # {2: 2, 3: 1}

# 約数列挙
def make_divisors(n):
    lower_divisors , upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]

    make_divisors(6)            # [1, 2, 3, 6]

# 素数判定
# O(√n)
# https://qiita.com/takayg1/items/3769ab4cc62a231f4259
def is_prime(n):
    if n == 1:
        return True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 素数列挙(エラトステネスの篩)
# O(NloglogN)
def eratosthenes_sieve(n):
    is_prime = [True]*(n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, n + 1):
        if is_prime[p]:
            for q in range(2*p, n + 1, p):
                is_prime[q] = False
    return is_prime
def use_eratosthenes():
    n = 10
    is_prime = eratosthenes_sieve(n)
    print(is_prime[5]) # True
    print(is_prime[10]) # False



if __name__ == '__main__':
    solve()

