
class ThreadingTask(object):
    """
    Defines a task
    """

    def __init__(self, code, extra_params={}):
        self.code = code
        self.extra_params = extra_params

class LearnerResult(object):

    def __init__(self, policy, stats=None):
        self.policy = policy
        self.stats = stats

class SamplingResult(object):

    def __init__(self, path):
        self.path = path

class SamplingTask(ThreadingTask):
    KILL_CODE = 0
    PUT_PARAMS_CODE = 1
    GET_PARAMS_CODE = 2
    COLLECT_SAMPLES_CODE = 3
    FIXED_SAMPLES_ROLLOUT_CODE = 4
    SET_EXTERNAL_POLICY_VALUES = 5
    REBUILD_NET = 6
    SET_ENV_TASK = 7

class SetExternalPolicyValuesTask(SamplingTask):
    def __init__(self, weights):
        LearnerTask.__init__(self, SamplingTask.SET_EXTERNAL_POLICY_VALUES, {"weights" : weights})

class SetParamsTask(SamplingTask):

    def __init__(self, params):
        SamplingTask.__init__(self, SamplingTask.PUT_PARAMS_CODE, {"policy" : params })

class RunFixedSamplesRolloutTask(SamplingTask):

    def __init__(self, num_samples):
        SamplingTask.__init__(self, SamplingTask.FIXED_SAMPLES_ROLLOUT_CODE, {"num_samples":num_samples})

class SetEnvTask(SamplingTask):

    def __init__(self, env):
        SamplingTask.__init__(self, SamplingTask.SET_ENV_TASK, {"env":env})

class RunRolloutTask(SamplingTask):

    def __init__(self):
        SamplingTask.__init__(self, SamplingTask.COLLECT_SAMPLES_CODE)

class KillThreadTask(SamplingTask):
    def __init__(self):
        SamplingTask.__init__(self, SamplingTask.KILL_CODE)

class LearnerTask(ThreadingTask):
    KILL_CODE = 0
    PUT_PARAMS_CODE = 1
    GET_PARAMS_CODE = 2
    ADJUST_MAX_KL = 3
    LEARN_PATHS = 4
    SET_EXTERNAL_POLICY_VALUES = 5
    REBUILD_NET = 6
    SET_ENV_TASK = 7




class RebuildNetSampleTask(SamplingTask):
    def __init__(self, **kwargs):
        SamplingTask.__init__(self, SamplingTask.REBUILD_NET, kwargs)

class RebuildNetTask(LearnerTask):
    def __init__(self, **kwargs):
        LearnerTask.__init__(self, LearnerTask.REBUILD_NET, kwargs)


class PutPolicyParamsTask(LearnerTask):
    def __init__(self, weights):
        LearnerTask.__init__(self, LearnerTask.PUT_PARAMS_CODE, {"weights" : weights})

class SetExternalPolicyValuesTask(LearnerTask):
    def __init__(self, weights):
        LearnerTask.__init__(self, LearnerTask.SET_EXTERNAL_POLICY_VALUES, {"weights" : weights})

class GetParamsTask(LearnerTask):

    def __init__(self):
        LearnerTask.__init__(self, LearnerTask.GET_PARAMS_CODE)

class AdjustMaxKLTask(LearnerTask):

    def __init__(self, max_kl):
        LearnerTask.__init__(self, LearnerTask.ADJUST_MAX_KL, {"max_kl" : max_kl})

class LearnFromPathsTask(LearnerTask):

    def __init__(self, paths):
        LearnerTask.__init__(self, LearnerTask.LEARN_PATHS, {"paths" : paths})
