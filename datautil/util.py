def Nmax(test_envs, d):
    for i in range(len(test_envs)):
        if d < test_envs[i]:
            return i
    return len(test_envs)