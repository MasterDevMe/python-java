import sys
import zmq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
from scipy.interpolate import griddata
from scipy.stats import norm
from copy import deepcopy

def solve(s1, s2, coefs):
    a, b, c, d, f, g, k, m = coefs
    A = a*k - c*f
    B = b*k + a*(m-s2) - c*g - f*(d-s1)
    C = b*(m-s2) - g*(d-s1)

    if (B**2.0 - 4.0*A*C > 0.0):
        res = (-B/(2*A) + np.sqrt(B**2.0 - 4.0*A*C)/(2*A),
                -B/(2*A) - np.sqrt(B**2.0 - 4.0*A*C)/(2*A))
        return res

def removeOutliers(S1):

    n = 5
    nsig = 3.0
    So = deepcopy(S1)

    for i in range(len(S1)):

        if i < n/2:
            # So.append(S1[i])
            pass
        elif i >= len(S1)-n/2:
            # So.append(S1[i])
            pass
        else:
            s = list(S1[i:i+n])
            val = s.pop(n/2)
            m = np.mean(s)
            std = np.std(s)
            if abs(val-m) > nsig*std:
                So[i] = m
            else:
                So[i] = val

    return So




class SCLI:

    def __init__(self, dataport=5556, ccport=5555):
        self.commands = ['list', 'help', 'tare', 'fit', 'cal', 'response', 'save', 'load', 'weigh', 'cont', 'stat', 'clear', 'plot', 'write',
         'plotresp', 'quit']
        self.cmd = None
        self.opts = None
        ctx = zmq.Context()
        self.dataport = dataport
        self.datasocket = ctx.socket(zmq.SUB)
        self.caldata = {}
        self.caldata['data'] = {}
        self.cmdsocket = ctx.socket(zmq.PUB)
        self.cmdsocket.bind("tcp://*:{}".format(ccport))

    def load_command(self, cmdstr):
        cs = cmdstr.split()
        self.cmd = cs[0].lower()
        if len(cs) > 1:
            self.opts = ' '.join(cs[1:])
        else:
            self.opts = None

    def parse_options(self, opts):

        res = {}
        os = opts.split()

        for item in os:
            os2 = item.split('=')

            if len(os2) != 2:
                print('WARNING: Option parse error for {}'.format(item))
                return
            else:
                res[os2[0]] = os2[1]
        return res

    def listcommands(self):
        for cmd in self.commands:
            print(cmd)

    def help(self, odict=None):

        cmds = self.commands
        if odict is not None:
            if 'command' in odict:
                if odict['command'] in cmds:
                    cmds = [odict['command']]
                else:
                    print('{} is not an available command\n')


        for c in cmds:
            if c == 'help':
                print('Command: help command=<command>')
                print('help on a specific <command> or all commands\n')

            if c == 'tare':
                print('Command: tare')
                print('Provide the tare weight reset to the device\n')

            if c == 'fit':
                print('Command: fit')
                print('Reperforms calibration fit for data within the current cal data (should not be needed)\n')

            if c == 'cal':
                print('Command: cal weight=<weight> npos=<linear position count>')
                print('The cal command starts a linear calibration for a single <weight> consisting of measurements along <npos> positions\n')

            if c == 'response':
                print('Command: response weight=<weight>')
                print('The response command starts a spatial response data collection for the weight <weight>\n')

            if c == 'save':
                print('Command: save file=<calibration filename>')
                print('Save the current data to file <filename>\n')

            if c == 'load':
                print('Command: load file=< calibration filename>')
                print('Load the data from <calibration filename>\n')

            if c == 'weigh':
                print('Command: weigh tint=<integration time>')
                print('Perform a weighing evolution\n')

            if c == 'cont':
                print('Command: cont time=<length of time> tint=<integration time>')
                print('Continuous weight testing with integration time for a length of time\n')

            if c == 'stat':
                print('Command: stat tint=<integration time>')
                print('Continuous weight based on a single reference while plotting probability distribution.  '
                      'Closing plot window exits this mode.\n')

            if c == 'clear':
                print('Command: clear')
                print('Clear the current calibration data.\n')

            if c == 'plot':
                print('Command: plot')
                print('Plot the current calibration plots.\n')

            if c == 'write':
                print('Command: write file=<coef csv filename>')
                print('Writes the current calibration coefs to file < filename>\n')

            if c == 'quit':
                print('Command: quit')
                print('Quit the program.\n')



    def statistical_weight(self, sampletime=1.0):

        if 'cal' in self.caldata:
            coefs = self.caldata['cal']
        else:
            print('ERROR: Calibration not updated...')
            return


        print('Setting reference...')
        time.sleep(2.0)
        ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=5.0)
        s1ref = np.mean(removeOutliers(s1))
        s2ref = np.mean(removeOutliers(s2))

        res = raw_input('Place Weight and <Enter>')
        if res == 'q':
            return
        print('Starting...')

        t1 = time.time()
        dt = time.time() - t1

        wt = []
        plt.ion()
        fig, ax = plt.subplots(1, 1)

        while plt.get_fignums():
            ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=sampletime, disp=False)
            ds1 = (np.mean(s1) - s1ref)
            ds2 = (np.mean(s2) - s2ref)
            wt_data = max(solve(ds1, ds2, coefs))
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")
            sys.stdout.write('Weight: {} g.'.format(wt_data))
            sys.stdout.flush()
            dt = time.time() - t1
            wt.append(wt_data)
            wtarr = np.array(wt)
            wt_mn = np.mean(wtarr)
            n,bins = np.histogram(wtarr, bins=20)
            bc = (bins[:-1] + bins[1:])/2.0
            nnorm = np.array(n).astype(float)/sum(n)/(bins[1]-bins[0])

            ax.cla()
            ax.plot(bc, nnorm, 'k.')
            if len(wt) > 20:
                mu, std = norm.fit(wt)
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'k', linewidth=2)
                title = "Results: mu = %.2f,  std = %.2f" % (mu, std)
                plt.title(title)

            ax.set_xlabel('Weight (grams)')
            ax.set_ylabel('probability per {} g.'.format(bins[1]-bins[0]))
            plt.pause(0.01)

        print('')


    def continuous_weight(self, twin=None, inttime=None):

        if twin is None:
            twin = 10.0

        if inttime is None:
            inttime = 2.0

        if 'cal' in self.caldata:
            coefs = self.caldata['cal']
        else:
            print('ERROR: Calibration not updated...')
            return

        res = raw_input('Set Weight Reference and <Enter> (q to quit)')
        if res == 'q':
            return

        ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=inttime)
        s1ref = np.mean(removeOutliers(s1))
        s2ref = np.mean(removeOutliers(s2))

        res = raw_input('Ready? <Enter> (q to quit)')
        if res == 'q':
            return

        t1 = time.time()
        dt = time.time()-t1

        while(dt < twin):
            ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=inttime, disp=False)
            ds1 = (np.mean(removeOutliers(s1)) - s1ref)
            ds2 = (np.mean(removeOutliers(s2)) - s2ref)
            wt = max(solve(ds1, ds2, coefs))
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")
            sys.stdout.write('Weight: {} g.'.format(wt))
            sys.stdout.flush()
            dt = time.time()-t1

        print('')


    def weight_test(self, tint):

        if 'cal' in self.caldata:
            coefs = self.caldata['cal']
        else:
            print('ERROR: Calibration not updated...')
            return

        if tint is None:
            tint = 5.0

        res = raw_input('Weight Reference and <Enter> (q to quit)')
        if res == 'q':
            return

        time.sleep(2.0)

        ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=tint)
        s1ref = np.mean(removeOutliers(s1))
        s2ref = np.mean(removeOutliers(s2))

        res = raw_input('Place weight and <Enter> (q to quit)')
        if res == 'q':
            return

        ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=tint)

        ds1 = (np.mean(removeOutliers(s1)) - s1ref)
        ds2 = (np.mean(removeOutliers(s2)) - s2ref)

        wt = max(solve(ds1, ds2, coefs))
        print('DS1: {} , DS2: {}'.format(ds1,ds2))
        print('Computed Weight: {} grams'.format(wt))

    def plotWeightResponse(self, wt):

        if wt not in self.caldata['data']:
            print('ERROR: No data for {} grams in calibration file loaded'.format(wt))
            k = []
            for key in self.caldata['data']:
                if 'response' in self.caldata['data'][key]:
                    k.append(key)
            print('Response data available for: {}'.format(k))
            return

        if 'response' not in self.caldata['data'][wt]:
            print('ERROR: No spatial response data for {} grams in calibration file loaded!'.format(wt))
            return

        x = self.caldata['data'][wt]['response'][0]
        y = self.caldata['data'][wt]['response'][1]
        d1 = self.caldata['data'][wt]['response'][2]
        d2 = self.caldata['data'][wt]['response'][3]

        xi = np.linspace(-3.0, 3.0, 50)
        yi = np.linspace(-3.0,3.0, 50)
        z1 = griddata((x, y), d1, (xi[None, :], yi[:, None]), method='linear')
        z2 = griddata((x, y), d2, (xi[None, :], yi[:, None]), method='linear')
        Xi, Yi = np.meshgrid(xi, yi)

        fig, (ax1,ax2) = plt.subplots(1,2)
        ct1 = ax1.contour(Xi,Yi,z1)
        ax1.clabel(ct1, inline=1, fontsize=10)
        c1 = plt.Circle((0, 0), 3.25, color='r', alpha=0.5)
        r1 = plt.Rectangle((-0.5,-3.25-4.0),1.0,4.0,color='k',alpha=0.5)
        ax1.add_artist(c1)
        ax1.add_artist(r1)
        ax1.axis('equal')
        ax1.set_xlim(-4,4)
        ax1.set_ylim(-8,5)
        ax1.set_title('{}g: Signal 1 Response'.format(wt))

        ct2 = ax2.contour(Xi,Yi,z2)
        ax2.clabel(ct2, inline=1, fontsize=10)
        ax2.clabel(ct1, inline=1, fontsize=10)
        c2 = plt.Circle((0, 0), 3.25, color='r', alpha=0.5)
        r2 = plt.Rectangle((-0.5,-3.25-4.0),1.0,4.0,color='k',alpha=0.5)
        ax2.add_artist(c2)
        ax2.add_artist(r2)

        ax2.axis('equal')
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-8, 5)
        ax2.set_title('{}g: Signal 2 Response'.format(wt))

        plt.show()

    def weight_response(self, odict):

        if 'weight' in odict:
            try:
                odict['weight'] = float(odict['weight'])
            except:
                odict.pop('weight')


        if 'weight' in odict:

            wt = odict['weight']

            self.datasocket.connect("tcp://localhost:{}".format(self.dataport))
            self.datasocket.setsockopt(zmq.SUBSCRIBE, "10000")

            mapping = [
                        (-1,-3),(0,-3),(1,-3),
                        (-2, -2), (-1, -2), (0, -2),(1, -2), (2, -2),
                        (-3,-1),(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),(3,-1),
                        (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
                        (-3, 1), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1),
                        (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
                        (-1, 3), (0, 3), (1, 3)
                       ]

            # Calibration Procedure
            x = []
            y = []
            ds1 = []
            ds2 = []
            for i in range(len(mapping)):
                res = raw_input('Remove all weight and <Enter>')
                if res == 'q':
                    break
                time.sleep(2.0)

                ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=3.0)
                s1ref = np.mean(removeOutliers(s1))
                s2ref = np.mean(removeOutliers(s2))

                res = raw_input('Place {}g weight at position ({},{}) and <Enter>'.format(wt,mapping[i][0],mapping[i][1]))
                if res == 'q':
                    break

                ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=3.0)
                x.append(float(mapping[i][0]))
                y.append(float(mapping[i][1]))
                ds1.append(np.mean(removeOutliers(s1)) - s1ref)
                ds2.append(np.mean(removeOutliers(s2)) - s2ref)
                print('Position {}: s1: {}, s2: {}'.format(i,ds1[-1],ds2[-1]))

            if len(x) == len(mapping):
                if not wt in self.caldata['data']:
                    self.caldata['data'][wt] = {}
                self.caldata['data'][wt]['response'] = [x,y,ds1,ds2]



    def calibrate_weight(self, opts):

        odict = self.parse_options(opts)

        if 'weight' in odict:
            try:
                odict['weight'] = float(odict['weight'])
            except:
                odict.pop('weight')

        if 'npos' in odict:
            try:
                odict['npos'] = int(odict['npos'])
            except:
                odict.pop('npos')


        if 'weight' in odict and 'npos' in odict:

            wt = odict['weight']
            npos = odict['npos']

            self.datasocket.connect("tcp://localhost:{}".format(self.dataport))
            self.datasocket.setsockopt(zmq.SUBSCRIBE, "10000")

            # Calibration Procedure
            ds1 = []
            ds2 = []
            for i in range(1,npos+1):
                res = raw_input('Remove all weight and <Enter>')
                if res == 'q':
                    break
                time.sleep(2.0)

                ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=5.0)
                s1ref = np.mean(removeOutliers(s1))
                s2ref = np.mean(removeOutliers(s2))

                res = raw_input('Place {}g weight at position {} and <Enter>'.format(wt,i))
                if res == 'q':
                    break

                ts, s1, s1o, s2, s2o, temp = self.retrieve_data(time_window=5.0)

                ds1.append(np.mean(removeOutliers(s1)) - s1ref)
                ds2.append(np.mean(removeOutliers(s2)) - s2ref)
                print('Position {}: s1: {}, s2: {}'.format(i,ds1[-1],ds2[-1]))

            if len(ds1) == npos:
                print('Calibration for Weight: {}g'.format(wt))
                print('Pos#\tS1\tS2')
                for i in range(npos):
                    print('{}\t{}\t{}'.format(i,ds1[i],ds2[i]))
                res = raw_input('Accept? (Y/n): ')
                if res == 'Y':
                    self.caldata['data'][wt] = {}
                    self.caldata['data'][wt]['ds1'] = ds1
                    self.caldata['data'][wt]['ds2'] = ds2
                    self.posFit()
                    self.wtFit()

    def retrieve_data(self, time_window=0.0, disp=True):
        self.datasocket.connect("tcp://localhost:{}".format(self.dataport))
        self.datasocket.setsockopt(zmq.SUBSCRIBE, "10000")
        #flush cache
        t1 = time.time()
        while(time.time()-t1 < 0.1):
            t1 = time.time()
            d = self.datasocket.recv()
            # topic, data = d.split()
            # ts, s1, s1offset, s2, s2offset, temp = data.split(',')


        TS, S1, S1OS, S2, S2OS, TEMP = [], [], [], [], [], []
        while(True):
            if disp:
                sys.stdout.write('.')
                sys.stdout.flush()
            d = self.datasocket.recv()
            topic, data = d.split()
            ts, s1, s1offset, s2, s2offset, temp = data.split(',')
            TS.append(float(ts))
            S1.append(float(s1))
            S1OS.append(float(s1offset))
            S2.append(float(s2))
            S2OS.append(float(s2offset))
            TEMP.append(float(temp))

            if TS[-1]-t1 > time_window:
                break
        sys.stdout.write('\n')
        self.datasocket.disconnect("tcp://localhost:{}".format(self.dataport))
        return TS, S1, S1OS, S2, S2OS, TEMP

    def posFit(self):

        for wt in self.caldata['data']:
            CD = self.caldata['data'][wt]

            if 'ds1' in CD and 'ds2' in CD:
                d1 = np.array(CD['ds1'])
                d2 = np.array(CD['ds2'])
                pos = np.array(range(len(d1)))

                if len(pos) > 1:
                    p1 = np.polyfit(pos, d1, 1)
                    p2 = np.polyfit(pos, d2, 1)
                    CD['p1'] = p1
                    CD['p2'] = p2

    def wtFit(self):

        wts = self.caldata['data'].keys()

        if len(wts) > 1:
            self.posFit()  # Just in case
            S1_m = []
            S1_b = []
            S2_m = []
            S2_b = []

            for wt in wts:
                cd = self.caldata['data'][wt]
                S1_m.append(cd['p1'][0])
                S1_b.append(cd['p1'][1])
                S2_m.append(cd['p2'][0])
                S2_b.append(cd['p2'][1])

            a,b = np.polyfit(wts, S1_m, 1)
            c,d = np.polyfit(wts, S1_b, 1)
            f,g = np.polyfit(wts, S2_m, 1)
            k,m = np.polyfit(wts, S2_b, 1)

            self.caldata['cal'] = (a,b,c,d,f,g,k,m)


    def plotResults(self):

        clr = ['b','r','g','k','m','y','c']

        fig, (ax1, ax2) = plt.subplots(1,2)
        i = 0

        for wt in self.caldata['data']:
            cd = self.caldata['data'][wt]
            n = np.array(range(len(cd['ds1'])))
            ax1.plot(n, cd['ds1'], '{}o'.format(clr[i]), label='{}.s1'.format(wt))
            ax1.plot(n, cd['ds2'], '{}s'.format(clr[i]), label='{}.s2'.format(wt))

            if 'p1' in cd:
                m,b = cd['p1']
                ax1.plot(n, b + m*n, '{}'.format(clr[i]))
            if 'p2' in cd:
                m,b = cd['p2']
                ax1.plot(n, b + m*n, '{}'.format(clr[i]))


            i+=1

        ax1.set_xlabel('Position')
        ax1.set_ylabel('Signal Output')
        ax1.set_title('Signal v. Position')

        ax1.grid(True)
        ax1.legend()

        S1_m = []
        S1_b = []
        S2_m = []
        S2_b = []
        wts = np.array(self.caldata['data'].keys())

        for wt in self.caldata['data']:
            cd = self.caldata['data'][wt]
            S1_m.append(cd['p1'][0])
            S1_b.append(cd['p1'][1])
            S2_m.append(cd['p2'][0])
            S2_b.append(cd['p2'][1])

        ax2.plot(wts, S1_m,'bo',label='S1_m')
        ax2.plot(wts, S1_b,'bs',label='S1_b')
        ax2.plot(wts, S2_m,'ro',label='S2_m')
        ax2.plot(wts, S2_b,'rs',label='S2_b')

        if 'cal' in self.caldata:
            a,b,c,d,f,g,k,m = self.caldata['cal']
            ax2.plot(wts, a*wts+b,'b')
            ax2.plot(wts, c*wts+d,'b')
            ax2.plot(wts, f*wts+g,'r')
            ax2.plot(wts, k*wts+m,'r')

        ax2.set_xlabel('Weight (g)')
        ax2.set_ylabel('Fit parameters')
        ax2.set_title('Fitting Parameters Trend')
        ax2.grid(True)
        ax2.legend()

        plt.show()

    def write(self, fname):

        if 'cal' in self.caldata:
            coefs = self.caldata['cal']
            try:
                with open(fname, 'w') as cfile:
                    s = '{},{},{},{},{},{},{},{}\n'.format(
                        coefs[0],coefs[1],coefs[2],coefs[3],
                        coefs[4],coefs[5],coefs[6],coefs[7]
                    )
                    cfile.write(s)
            except:
                print('ERROR: Problem saving coefficients to {}'.format(fname))
                return


    def dispatch(self):

        odict = None
        if self.opts is not None:
            if len(self.opts) > 0:
                odict = self.parse_options(self.opts)

        if self.cmd == 'help':
            self.help(odict)

        if self.cmd == 'tare':
            self.cmdsocket.send("1 t")

        if self.cmd == 'fit':
            self.posFit()
            self.wtFit()

        if self.cmd == 'cal' or self.cmd=='c':
            self.calibrate_weight(self.opts)

        if self.cmd == 'response' or self.cmd=='r':
            if odict is not None:
                if 'weight' in odict:
                    self.weight_response(odict)
                else:
                    print('ERROR: response requires <weight>')

        if self.cmd == 'save':

            if odict is not None:
                if 'file' in odict:
                    try:
                        pickle.dump(self.caldata, open(odict['file'], "wb"))
                    except:
                        print('ERROR: Problem saving to {}'.format(odict['file']))
                        return
                else:
                    print('save requires "file=<filename>" option')

        if self.cmd == 'weigh':

            tint = None
            if odict is not None:
                if 'tint' in odict:
                    try:
                        tint = float(odict['tint'])
                    except:
                        tint = None
                        print('ERROR converting {} to numeric value for integration time...using default values')
            self.weight_test(tint)

        if self.cmd == 'cont':

            twin=None
            tint = None
            if odict is not None:
                if 'time' in odict:
                    twin = float(odict['time'])
                if 'tint' in odict:
                    tint = float(odict['tint'])
            self.continuous_weight(twin=twin, inttime=tint)

        if self.cmd == 'stat':

            sampleTime = 1.0
            if odict is not None:
                if 'tint' in odict:
                    sampleTime = float(odict['tint'])

            self.statistical_weight(sampletime=sampleTime)


        if self.cmd == 'clear':
            self.caldata['data'] = {}

        if self.cmd == 'load':
            if odict is not None:
                if 'file' in odict:
                    try:
                        self.caldata = pickle.load(open(odict['file'], "rb"))
                    except:
                        print('ERROR: Problem opening {}'.format(odict['file']))
                        return
                else:
                    print('load requires "file=<filename>" option')

        if self.cmd == 'plot':
            self.plotResults()

        if self.cmd == 'list':
            self.listcommands()

        if self.cmd == 'write':

            if odict is not None:
                if 'file' in odict:
                    self.write(odict['file'])
                else:
                    print('write requires "file=<filename>" option')

        if self.cmd == 'plotresp':
            if odict is not None:
                if 'weight' in odict:
                    try:
                        wf = float(odict['weight'])
                    except:
                        print('ERROR: Cannot convert {} to a weight!'.format(odict['weight']))
                        return

                    self.plotWeightResponse(float(odict['weight']))
                else:
                    print('plotresp requires "weight=<weight>" option')
            else:
                print('"weight" option is required')


    def run(self):

        print('\n\n***** S Command Line Interface *****\n')
        while self.cmd != "quit":

            res = raw_input("S>> ")
            self.load_command(res)
            self.dispatch()


        print('Exiting S Command Line Interface...Bye!')


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    CLI = SCLI()
    CLI.run()
