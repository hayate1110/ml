import random

def ga(N: int, M: int, n: int, N_e: int, P_m: float, P_c: float, sizes: list, values: list, limit: int):
    individuals = []
    N_c = N - N_e
    
    for i in range(N):
        genes = [random.randint(0, 3) for i in range(n)]
        individuals.append({"genes": genes, "fitness": 0})
    
    for i in range(M):
        elites = []
        children = []

        sum_f = 0
        for individual in individuals:
            individual["fitness"] = f(individual["genes"], sizes, values, limit, n)
            sum_f += individual["fitness"]

        individuals = sorted(individuals, key=lambda x: x["fitness"], reverse=True)
        
        print(f"{i}世代目:")
        print(f'    遺伝子列={individuals[0]["genes"]}')
        print(f'    適合度={individuals[0]["fitness"]}')
        print()

        for j in range(N_e):
            elites.append(individuals[j])
            sum_f -= individuals[j]["fitness"]
    
        for j in range(N_c // 2):
            parent1 = {}
            a = random.random()
            for k in range(N_e, N):
                parent1 = individuals[k]
                p = individual["fitness"] / sum_f
                if a < p:
                    break
                else:
                    a -= p

            parent2 = {}
            a = random.random()
            for k in range(N_e, N):
                parent2 = individuals[k]
                p = individual["fitness"] / sum_f
                if a < p:
                    break
                else:
                    a -= p

            a = random.random()
            if a < P_c:
                split_point = int(random.random() * n)

                children.append({"genes": parent1["genes"][:split_point] + parent2["genes"][split_point:], "fitness": 0})
                children.append({"genes": parent2["genes"][:split_point] + parent1["genes"][split_point:], "fitness": 0})
            else:
                children.append(parent1)
                children.append(parent2)

        for j in range(N_c-1):
            for k in range(n):
                a = random.random()
                if a < P_m:
                    dgene = random.randint(-3, 3)
                    children[j]["genes"][k] += dgene
                    if children[j]["genes"][k] < 0: 
                        children[j]["genes"][k] = 0

        individuals = children + elites

    return sorted(individuals, key=lambda x: x["fitness"], reverse=True)
            
def f(genes: list, sizes: list, values: list, limit: int, n: int):
    sum_size = 0
    sum_value = 0
    for i in range(n):
        count = genes[i]
        sum_size += sizes[i] * count
        sum_value += values[i] * count

    if sum_size > limit:
        sum_value = 0
    
    return sum_value

if __name__ == "__main__":
    N = 30
    M = 100
    n = 5
    N_e = 3
    P_m = 0.005
    P_c = 0.9
    sizes = [20, 10, 5, 15, 3]
    values = [80, 30, 20, 25, 10]
    limit = 200
    individuals = ga(N, M, n, N_e, P_m, P_c, sizes, values, limit)
                
    