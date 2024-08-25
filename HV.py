import numpy as np


def HV(PopObj, PF):
    N, M = PopObj.shape
    RefPoint = np.max(PF, axis=0) * 1.1
    PopObj = PopObj[~np.any(PopObj > np.tile(RefPoint, (N, 1)), axis=1), :]

    if PopObj.size == 0:
        Score = 0
    elif M < 5:
        pl = np.sort(PopObj, axis=0)
        S = [(1, pl)]

        for k in range(1, M):
            S_ = []

            for i in range(len(S)):
                Stemp = Slice(S[i][1], k, RefPoint)

                for j in range(Stemp.shape[0]):
                    temp = (S[i][0] * Stemp[j, 0], Stemp[j, 1])
                    S_ = Add(temp, S_)

            S = S_

        Score = 0

        for i in range(len(S)):
            p = Head(S[i][1])
            Score += S[i][0] * abs(p[-1] - RefPoint[-1])

    else:
        SampleNum = 1000000
        MaxValue = RefPoint
        MinValue = np.min(PopObj, axis=0)
        Samples = np.tile(MinValue, (SampleNum, 1)) + np.random.rand(SampleNum, M) * np.tile((MaxValue - MinValue),
                                                                                             (SampleNum, 1))
        Domi = np.zeros(SampleNum, dtype=bool)

        for i in range(PopObj.shape[0]):
            Domi[np.all(np.tile(PopObj[i, :], (SampleNum, 1)) <= Samples, axis=1)] = True

        Score = np.prod(MaxValue - MinValue) * np.sum(Domi) / SampleNum

    return Score


def Slice(PopObj, k, RefPoint):
    N, M = PopObj.shape
    Pl = np.sort(PopObj[:, k])
    plen = len(Pl)
    Seq = np.ceil(np.log2(plen))
    SectLen = np.zeros(Seq)
    SectLen[0] = 2 ** (Seq - 1)

    for i in range(1, Seq):
        SectLen[i] = SectLen[i - 1] / 2

    SectLen = SectLen.astype(int)
    Section = np.zeros((Seq, M + 1))
    Section[0, :] = np.append(-np.inf, Pl[SectLen[0] - 1])

    for i in range(1, Seq):
        Section[i, :] = np.append(Section[i - 1, 1], Pl[SectLen[i] + SectLen[i - 1] - 1])

    Section[Seq - 1, -1] = np.inf
    sliceSet = np.zeros((plen, 2), dtype=object)
    S = 0

    for i in range(Seq):
        for j in range(S, min(S + SectLen[i], plen)):
            sliceSet[j, :] = (Section[i, :-1], np.append(Section[i, 1:], RefPoint[k]))

        S += SectLen[i]

    return sliceSet


def Head(PopObj):
    N = PopObj.shape[0]
    L = np.zeros(N, dtype=bool)

    for i
