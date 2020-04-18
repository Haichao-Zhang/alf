# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sampling Determinantal point processes (DPPs) from the
#   sequential thinning algorithm
#
# NB: You can find a short script to test this algorithm at the end
#       of the file.
#
# Input:
# - K: any DPP kernel, an hermitian matrix with
#       eigenvalues between 0 and 1
# [- q: the associated Bernoulli probabilities,
#       from the function sequential_thinning_init, optional]
#
# Output:
# - A: DPP realization of kernel K
#
# C. Launay, B. Galerne, A. Desolneux (c) 2018

import torch

#print(mydevice)
#torch.set_default_tensor_type(torch.DoubleTensor)


def sequential_thinning_dpp_simulation(K, X):
    # if q.size(0) == 0:
    #     q = sequential_thinning_dpp_init(K)

    # # Draw dominating Bernouilli process:
    # X = (torch.rand(q.size()) < q).nonzero()
    # print(X)

    # Initialization:
    A = torch.tensor([], dtype=torch.long)  # Indices of selected points
    B = torch.tensor([], dtype=torch.long)  # Indices of unseclected points
    NB = torch.tensor([],
                      dtype=torch.long)  # New indices of B between points of X
    L = torch.tensor([])  # Cholesky decomposition of (I-K)_B
    ImK = torch.eye(K.size(0)) - K

    previous_point = -1
    for k in X:

        # update list of new point not in  Y since previous point of X:
        # (before that NB is either [] or previous_point)
        NB = torch.cat((NB,
                        torch.arange(
                            previous_point + 1, int(k), dtype=torch.long)))
        if len(NB) > 0:
            if len(B) == 0:
                L = torch.cholesky(ImK[NB, :][:, NB], upper=False)
            else:
                L = cholesky_update_add_bloc(
                    L, ImK[B, :][:, NB].view(len(B), len(NB)),
                    ImK[NB, :][:, NB].view(len(NB), len(NB)))
        # update B:
        B = torch.cat((B, NB))
        Aandk = torch.cat((A, torch.tensor([k], dtype=torch.long)))
        if len(NB) == 0:
            H = K[Aandk, :][:, Aandk]
        else:
            J = torch.triangular_solve(K[B, :][:, Aandk], L, upper=False)[0]
            H = K[Aandk, :][:, Aandk] + torch.mm(J.t(), J)
        if len(A) == 0:
            pk = H
        else:
            LH = torch.cholesky(H[:-1, :-1], upper=False)
            pk = H[-1, -1] - torch.sum(
                torch.triangular_solve(H[:-1, [-1]], LH, upper=False)[0]**2)
        print(pk)
        if torch.rand(1) < pk:
            # add k to A:
            A = torch.cat((A, torch.tensor([k], dtype=torch.long)))
            NB = torch.tensor([], dtype=torch.long)
        else:
            # add k to NB (to be included in B at next iteration):
            NB = torch.tensor([k], dtype=torch.long)
        previous_point = int(k)
    return (A)


def sequential_thinning_dpp_init(K):
    N = K.size(0)
    try:
        L = torch.cholesky(torch.eye(N - 1) - K[:-1, :-1], upper=False)
        B = torch.triu(K[0:-1, 1:])
        q = K.diag()
        q[1:].add_(
            torch.sum(
                torch.triu(torch.triangular_solve(B, L, upper=False)[0])**2,
                0))
    except RuntimeError:  #slower procedure if I-K is singular
        q = torch.ones([N])
        ImK = torch.eye(K.size(0)) - K
        q[0] = K[0, 0]
        for k in range(1, N):
            q[k] = K[k, k] + torch.mm(
                K[[k], :k], torch.cholesky_solve(K[:k, [k]], ImK[:k, :k]))
            if q[k] == 1:
                break
    return (q)


def cholesky_update_add_bloc(LA, B, C):
    if len(LA) == 0:
        LM = torch.cholesky(C, upper=False)
    else:
        V = torch.triangular_solve(B, LA, upper=False)[0]
        LX = torch.cholesky(C - torch.mm(V.t(), V), upper=False)
        LM = torch.cat([
            torch.cat([LA, torch.zeros(B.shape)], 1),
            torch.cat([V.t(), LX], 1)
        ], 0)
    return (LM)


# N = 5000
# EK = 20
# K = random_dpp(N, EK).to(mydevice)
# q = sequential_thinning_dpp_init(K)
# A = sequential_thinning_dpp_simulation(K, q)
