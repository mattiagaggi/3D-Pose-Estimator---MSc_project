import IPython
import torch.nn

def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None, add_prefix=''):

    own_state = obj.state_dict()
    copyCount = 0
    skipCount = 0
    paramCount = len(own_state)

    for name_raw, param in state_dict_other.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if prefix is not None and not name_raw.startswith(prefix):
            # print("skipping {} because of prefix {}".format(name_raw, prefix))
            continue

        # remove the path of the submodule from which we load
        name = add_prefix + ".".join(name_raw.split('.')[submodule:])
        if name in own_state:
            if hasattr(own_state[name], 'copy_'):  # isinstance(own_state[name], torch.Tensor):
                # print('copy_ ',name)
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                    copyCount += 1
                else:
                    #print('Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(),
                    #                                                                     param.size(), name))
                    skipCount += 1

            elif hasattr(own_state[name], 'copy'):
                own_state[name] = param.copy()
                copyCount += 1
            else:
                #print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name, name_raw))
                #print(type(own_state[name]))
                skipCount += 1
                IPython.embed()
        else:
            skipCount += 1
            #print('Warning, no match for {}, ignoring'.format(name))
            # print(' since own_state.keys() = ',own_state.keys())
    #print('Copied {} elements, {} skipped, and {} target params without source'.format(copyCount, skipCount,                                                                                   paramCount - copyCount))