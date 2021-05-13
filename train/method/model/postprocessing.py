import copy


def gets_height(outputs,position):
    '''def 求某一帧的 height'''# 输入siamese 的outputs 和所需要判断的帧的位置，输出帧的height列表
    height = []
    a = 0
    b = 0
    for i in (position):
        for j in range(i-1):
            if outputs[j,0]>0.5:
                a += 1
            else:
                b += 1
        height.append(a-b)
        a = 0
        b = 0
    return height


def gets_a4cd_frames(outputs_1):
    '''分别得到所有可能的a4cs a4cd'''
    a4cd = []
    for i in range(len(outputs_1)-1):
        if outputs_1[i,0]>0.5 and outputs_1[i+1,0]<0.5:  #a4cd
            a4cd.append(i+2)
    return a4cd


def gets_a4cs_frames(outputs_1):
    a4cs = []
    for i in range(len(outputs_1)-1):
        if outputs_1[i,0]<0.5 and outputs_1[i+1,0]>0.5:  #a4cs
            a4cs.append(i+2)
    return a4cs


def delete_a4cd_frames(outputs_1):
    a4cd = gets_a4cd_frames(outputs_1)
    a4cd_copy = copy.deepcopy(a4cd)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in (a4cd):
        t = i-2
        if t < 10:
            for j in range(t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            for j in range(t,t+10):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a>b and c<d):   #说明不满足 从a4cd中删除
                a4cd_copy.remove(i)
            a = 0
            b = 0
            c = 0
            d = 0
        elif t > len(outputs_1)-11:
            for j in range(t-9,t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            for j in range(t,len(outputs_1)):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a>b and c<d):   #说明不满足 从a4cd中删除
                a4cd_copy.remove(i)
            a = 0
            b = 0
            c = 0
            d = 0
        else:
            for j in range(t-9,t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            for j in range(t,t+10):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a>b and c<d):   #说明不满足 从a4cd中删除
                a4cd_copy.remove(i)

            a = 0
            b = 0
            c = 0
            d = 0

    return a4cd_copy


def delete_a4cs_frames(outputs_1):
    a4cs = gets_a4cs_frames(outputs_1)
    a4cs_copy = copy.deepcopy(a4cs)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in (a4cs):
        t = i-2
        if t < 10:
            for j in range(t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            for j in range(t,t+10):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a<b and c>d):   #说明不满足 从a4cd中删除
                a4cs_copy.remove(i)
            a = 0
            b = 0
            c = 0
            d = 0

            
            
            
        elif t > len(outputs_1)-11:
            for j in range(t-9,t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            for j in range(t,len(outputs_1)):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a<b and c>d):   #说明不满足 从a4cd中删除
                a4cs_copy.remove(i)
            a = 0
            b = 0
            c = 0
            d = 0

            
            
            
        else:
            for j in range(t-9,t+1):
                if outputs_1[j,0]>0.5:
                    a += 1
                else:
                    b += 1
            
                
                
            for j in range(t,t+10):
                if outputs_1[j,0]>0.5:
                    c += 1
                else:
                    d += 1
            if not (a<b and c>d):   #说明不满足 从a4cd中删除
                a4cs_copy.remove(i)

            a = 0
            b = 0
            c = 0
            d = 0
            
    
    return a4cs_copy
    
    
'''滑动窗口'''
def sliding_window(outputs,frames,key):
    '''
    outputs:siamese 输出的结果
    a4cd_frames:a4cd 帧列表
    key:a4cd or a4cs is 1 or 0
    滑动窗口大小为10，步长为1
    比较局部极值height，取max值，当存在相等时， 取最后一帧
    返回剩余帧列表

    '''
    i = 0
    j = 10
    Temp_list = []
    L1 = []
    Frames = []
    while True:
        for m in range(i,j):
            for n in frames:
                if m == n-2:
                    Temp_list.append(n)
        if len(Temp_list)>len(L1):
            L1 = Temp_list
            Temp_list = []
        else:
            Temp_list = []
        if (L1 and i == L1[-1] - 1) or (L1 and j == len(outputs)):
            L1.reverse()
            Height_list = gets_height(outputs,position=L1)
            if key:  #a4cd
                index = Height_list.index(max(Height_list))
                Frames.append(L1[index])
                L1 = []
            else:
                index = Height_list.index(min(Height_list))
                Frames.append(L1[index])
                L1 = []
        
    
        
        i += 1
        j += 1
        if j == len(outputs)+1:
            break
    return Frames
    
    
    
    





