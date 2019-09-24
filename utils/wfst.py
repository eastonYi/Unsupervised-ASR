import numpy as np
from time import time
import ctypes


def p(*info):
    print(*info)


def torch_c_ptr(thmat):
    state_ptr = thmat.data_ptr()
    return ctypes.c_void_p(state_ptr)


class cls_speed:

    def __init__(self):
        self.cdll = None
        self.check_type = 1 # 检查类型，我一开始不希望使用float32 后来为了速度才采用
        pass

    def load_cdll(self, fcdll):
        p("use dll", fcdll)
        self.cdll = ctypes.cdll.LoadLibrary(fcdll)

    def set_mp(self, nm, npmat):
        if self.check_type==1:
            if npmat.dtype==np.float32: p(nm,"type is float32")
            if npmat.dtype==np.int64: p(nm,"type is int64")

        pm = npmat.ctypes.data_as(ctypes.c_void_p)
        # bt=(ctypes.c_char * 100)().value
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, pm)

    # ctype type is ok,# numpy  may has problem
    def get_mp(self, nm, shape, ctype):
        #if(ctype==np.int32): ctype= types.c_int32
        ss = bytes(nm, encoding="ascii")
        #self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int32, shape=(3,4))
        self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctype, shape=shape)
        npmat=self.cdll.get_mp(ss)

        return npmat

    def set_mp_ext(self, nm, ctype_ptr):
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, ctype_ptr)

    # cpu/gpu都可以
    def set_mp_torch(self, nm, thmat):
        self.set_mp_ext(nm, torch_c_ptr(thmat))


class WFST_Decoder:

    def __init__(self, decode_outs, fcdll, fcfg):
        start = time()
        self.speed = cls_speed()
        self.speed.load_cdll(fcdll)
        self.speed.check_type=0
        fcfg_ = (ctypes.c_char * 100)().value = bytes(fcfg, encoding="ascii")
        self.speed.cdll.CHJ_CTC_LIB_init(fcfg_)
        self.len_decode_max = len(decode_outs)
        self.decode_outs = decode_outs
        self.speed.set_mp("decode_outs", self.decode_outs)
        print('loading wfst graph with {}s'.format(time() - start))

    def decode(self, distribution):

        distribution_log = np.log(distribution)
        dims = np.array([distribution_log.shape[0],
                         self.len_decode_max,
                         0,
                         0], dtype=np.int32)
        self.speed.set_mp("dims", dims)
        self.speed.set_mp("net_outs", distribution_log)
        self.speed.cdll.CHJ_CTC_LIB_run_one_sentence()
        len_decode = self.decode_outs[0]
        decoded = self.decode_outs[1: 1+len_decode]

        return decoded


if __name__ == '__main__':
    distribution = np.array([10, 512])

    # WFTS config
    len_decode_max=200
    decode_outs=np.zeros((len_decode_max), dtype=np.int32)
    wfst = WFST_Decoder(
        len_decode_max=len_decode_max,
        decode_outs=decode_outs,
        fcdll="bin/libctc_wfst_lib.so",
        fcfg="cfg.json")

    distribution_log = np.log(distribution)
    decoded = wfst.decode(distribution_log)
