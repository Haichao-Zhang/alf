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

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import copy


class BeamSearchNode(object):
    """The node is responsible for generating the actions for the transition
    from the current observation to the next one.
    """

    def __init__(self, previousNode, obs, prev_action, logProb, length):
        '''
        :param previousNode:
        :param obs: current observation/state
        # :param action: planned action based on the current observation
        :param prev_action: action leading to the current observation
        :param logProb:
        :param length:
        '''
        self.obs = obs
        self.prevNode = previousNode
        self.prev_action = prev_action  # for recovering the action sequence
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        return self.logp / float(self.leng - 1 + 1e-6)

    def __lt__(self, other):
        return self.eval() < other.eval()

    # def generate_children(self, ):
    #     """
    #     Implement: generating children
    #     """
    #     children = [Node(self.value+random.randint(-5,5)) for _ in xrange(3)]
    #     return children


def beam_decode(time_step,
                state,
                eval_func,
                dynamics_func,
                max_len=25,
                number_required=100,
                lower_bound=-2.,
                upper_bound=2.,
                solution_size=1):
    """ beam decode
    Args:
        time_step: input tensor of shape [B, H] for start of the decoding, where H
            is the state vector dimension
        TODO: might also need dynamics model
    Returns:
        decoded_batch
    """

    #number_required = pop_size
    beam_width = 10
    pop_size = 10 * beam_width  # expand size, will select beam_width from pop_size
    topk = beam_width  # how many sentence do you want to generate
    decoded_batch = []

    s0 = time_step.observation

    batch_size = s0.shape[0]
    # decoding goes element by element in batch
    for idx in range(batch_size):
        # keep batch dim
        s0_i = s0[idx, :]  # individual state in batch s0

        # Number of sentence to generate
        endnodes = []

        # starting node -  previous node, obs, prev_action, logp, length
        node = BeamSearchNode(None, s0_i, None, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            # if qsize > 2000: break

            # fetch the best nodegit
            score, n = nodes.get()
            current_obs = n.obs.unsqueeze(0)  # recover batch dim

            # +1 due to starting from the current step
            if n.leng == max_len + 1 and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # using action sampling and dynamics model to rollout onestep
            # decode for one step using decoder
            # batch size is 1
            ac_rand_pop = torch.rand(
                current_obs.shape[0] * pop_size,
                solution_size) * (upper_bound - lower_bound) + lower_bound

            obs_pop = torch.repeat_interleave(current_obs, pop_size, dim=0)
            critic_input = (obs_pop, ac_rand_pop)

            # critic, critic_state = self._policy_module._critic_network1(
            #     critic_input)

            critic, _ = eval_func(critic_input)  # both eval and dymanics

            time_step, state = dynamics_func(
                time_step._replace(prev_action=ac_rand_pop),
                state._replace(
                    dynamics=state.dynamics._replace(feature=obs_pop)))

            critic = critic.reshape(current_obs.shape[0], pop_size)

            sel_value, sel_ind = torch.topk(
                critic, k=min(beam_width, critic.shape[1]))

            ac_rand_pop = ac_rand_pop.reshape(current_obs.shape[0], pop_size,
                                              -1)
            next_obs_pop = time_step.observation.reshape(
                current_obs.shape[0], pop_size, -1)

            def _batched_index_select(t, dim, inds):
                dummy = inds.unsqueeze(2).expand(
                    inds.size(0), inds.size(1), t.size(2))
                out = t.gather(dim, dummy)  # b x e x f
                return out

            action = _batched_index_select(ac_rand_pop, 1, sel_ind).squeeze(1)
            next_obs = _batched_index_select(next_obs_pop, 1,
                                             sel_ind).squeeze(1)

            # the actual batch size is 1 as we iterate over elements in batch
            sel_value = sel_value.squeeze(0)
            action = action.squeeze(0)  #.reshape(-1, solution_size)
            next_obs = next_obs.squeeze(0)
            # PUT HERE REAL BEAM SEARCH OF TOP
            nextnodes = []

            for new_k in range(beam_width):
                log_p = sel_value[new_k]
                # branch node -  previous_node, obs, prev_action, logp, length
                node = BeamSearchNode(n, next_obs[new_k], action[new_k],
                                      n.logp + log_p.cpu().numpy(), n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.prev_action)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.prev_action)

            utterance = utterance[::-1]
            utterance_tensor = torch.cat(utterance[1:])
            # unsqueeze prepare for the outer cat
            utterances.append(utterance_tensor.unsqueeze(0))

        utterances_tensor = torch.cat(utterances, dim=0)
        decoded_batch.append(utterances_tensor.unsqueeze(0))

    decoded_batch_tensor = torch.cat(decoded_batch, dim=0)
    # [B, P, plan_horizon]
    return decoded_batch_tensor
