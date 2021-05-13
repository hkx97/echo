import copy


def gets_height(outputs,position):
    # volume-time curve:
    # from volume-time curve get any position point's heigth
    height = []
    a,b =0, 0
    for i in (position):
        for j in range(i-1):
            if outputs[j,0]>0.5:
                a += 1
            else:
                b += 1
        height.append(a-b)
        a ,b= 0,0
    return height


def gets_a4cd_frames(outputs_1):
    # get all candicate frames of inflection points
    # if outputs = [[0,1],[0,1],[0,1],[1,0],[1,0],[0,1]]
    #          *
    #       .    .    .
    #    .         *
    # .  
    # the * are candicate frames
    a4cd = []
    for i in range(len(outputs_1)-1):
        if outputs_1[i,0]>0.5 and outputs_1[i+1,0]<0.5:  #ED
            a4cd.append(i+2)
    return a4cd


def gets_a4cs_frames(outputs_1):
    a4cs = []
    for i in range(len(outputs_1)-1):
        if outputs_1[i,0]<0.5 and outputs_1[i+1,0]>0.5:  #ES
            a4cs.append(i+2)
    return a4cs


def delete_a4cd_frames(outputs_1):
    a4cd = gets_a4cd_frames(outputs_1)
    a4cd_copy = copy.deepcopy(a4cd)
    a,b,c,d= 0,0,0,0
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
            # if a candicate frame not satisfy the trend of up/down, it will be deleted.
            if not (a>b and c<d):   
                a4cd_copy.remove(i)
            a,b,c,d= 0,0,0,0
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
            if not (a>b and c<d):  
                a4cd_copy.remove(i)
            a,b,c,d= 0,0,0,0
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
            if not (a>b and c<d):   
                a4cd_copy.remove(i)
            a,b,c,d= 0,0,0,0
    return a4cd_copy


def delete_a4cs_frames(outputs_1):
    # a function for candicate frames selecting
    # outputs_1 is all candicate frames of inflection points
    a4cs = gets_a4cs_frames(outputs_1)
    a4cs_copy = copy.deepcopy(a4cs)
    a,b,c,d= 0,0,0,0
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
            if not (a<b and c>d):   
                a4cs_copy.remove(i)
            a,b,c,d= 0,0,0,0
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
            if not (a<b and c>d):   
                a4cs_copy.remove(i)
            a,b,c,d= 0,0,0,0
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
            if not (a<b and c>d):   
                a4cs_copy.remove(i)
            a,b,c,d= 0,0,0,0
    return a4cs_copy
    

def sliding_window(outputs,frames,key):
    '''
    outputs:model's output --> a n*2 tensor
    frames: a list of condicate frames
    key:ED or ES is 1 or 0
    sliding window : length is 10 frames , stride is 1 frame
    the frame with max/min heigth will return as ED/ES
    if >1frames have max heigth. the last one will be choiced.
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
    
    
    
    





