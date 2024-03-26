from typing import Union, List, Tuple, Iterable
class Formater:
    def __init__(self):
        self.EngineeringPrefixes = ['a', 'f', 'n', 'mk', 'm', '', 'K', 'M', 'G', 'T']
        self.EngineeringCenter = 5
        self.ScientificNumbers = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','⁻']

        DefaultStile = {
            'font': 'Times New Roman',
            'fontsize': 12,
            'fontweight': 'normal',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        HeaderStyle = {
            'font': 'Times New Roman',
            'fontsize': 16,
            'fontweight': 'bold',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        BigHeaderStyle = {
            'font': 'Times New Roman',
            'fontsize': 20,
            'fontweight': 'bold',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        CaptionStyle = {
            'font': 'Times New Roman',
            'fontsize': 8,
            'fontweight': 'normal',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        SmallCaptionStyle = {
            'font': 'Times New Roman',
            'fontsize': 6,
            'fontweight': 'light',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        self.Styles = {
            'Default': DefaultStile,
            'Header': HeaderStyle,
            'BigHeader': BigHeaderStyle,
            'Caption': CaptionStyle,
            'SmallCaption': SmallCaptionStyle
        }

    def Engineering(self, value, unit='', precision=3):
        if value is Iterable:
            String = '{'
            for val in value:
                String += self.Scientific(val, unit, precision) + ', '
            String = String[:-2] + '}'
            return String
        else:
            if value == 0:
                return str(round(value, precision)) + ' '
            c = self.EngineeringCenter
            sign = 1
            if value < 0:
                sign = -1
                value = -value
            while value <= 1.0 and c != 0:
                value *= 1000
                c -= 1
            while value >= 1000.0 and c != len(self.EngineeringPrefixes) - 1:
                value /= 1000
                c += 1
            return str(round(sign*value, precision)) + ' ' + self.EngineeringPrefixes[c] + unit
    def Engineering_Separated(self, value, unit=''):
        c = self.EngineeringCenter
        sign = 1
        if value == 0:
            return self.EngineeringPrefixes[c] + unit, 1000 ** (self.EngineeringCenter - c)
        if value < 0:
            sign = -1
            value = -value
        while value <= 1.0 and c != 0:
            value *= 1000
            c -= 1
        while value >= 1000.0 and c != len(self.EngineeringPrefixes) - 1:
            value /= 1000
            c += 1
        return self.EngineeringPrefixes[c] + unit, 1000**(self.EngineeringCenter-c)

    def Scientific(self, value, unit='', precision=3):
        if value is Iterable:
            String = '{'
            for val in value:
                String += self.Scientific(val, unit, precision) + ', '
            String = String[:-2] + '}'
            return String
        else:
            if value == 0:
                return '0 ' + unit
            c = 0
            sign = 1
            if (value < 0):
                sign = -1
                value = -value
            while value <= 1.0:
                value *= 10
                c -= 1
            while value >= 10.0:
                value /= 10
                c += 1
            ps = ''
            if c < 0:
                ps += self.ScientificNumbers[10]
                c = -c
            ps_ = ''
            if c == 0:
                return str(round(sign*value, precision)) + ' ' + unit
            while c != 0:
                ps_ += self.ScientificNumbers[int(c%10)]
                c = int(c/10)
            ps += ps_[::-1]
            return str(round(sign*value, precision)) + '·10' + ps + ' ' + unit
    def Scientific_Separated(self, value, unit=''):
        if value == 0:
            return '0 ' + unit
        c = 0
        sign = 1
        if (value < 0):
            sign = -1
            value = -value
        while value <= 1.0:
            value *= 10
            c -= 1
        while value >= 10.0:
            value /= 10
            c += 1
        power = c
        ps = ''
        if c < 0:
            ps += self.ScientificNumbers[10]
            c = -c
        ps_ = ''
        if c==0:
            return '1' + ' ' + unit, 1.0
        while c != 0:
            ps_ += self.ScientificNumbers[int(c%10)]
            c = int(c/10)
        ps += ps_[::-1]
        return '·10' + ps + ' ' + unit, 10**(-power)

    def Time(self, seconds=0, days=0, hours=0, minutes=0, millis=0, micros=0, nanos=0):
        time = seconds + millis*1.0E-3 + micros*1.0E-6 + nanos*1.0E-9
        days = int(time/86400)
        time -= days*86400
        hours = int(time/3600)
        time -= hours*3600
        minutes = int(time/60)
        time-= minutes*60
        seconds = int(time/1)
        time -= seconds*1
        millis = int(time*1000)
        time -= millis/1000
        micros = int(time*1000000)
        time -= micros/1000000
        nanos = int(time*1000000000)
        time -= nanos*1000000000
        time_array = [int(days),int(hours),int(minutes),int(seconds),int(millis),int(micros),int(nanos)]
        time_units = ['d','h','m','s','ms','us','ns']
        string = ''
        for i in range(len(time_array)):
            if time_array[i] != 0:
                string = str(time_array[i]) + time_units[i]
                if i != (len(time_array)-1) and time_array[i+1] != 0:
                    return string + str(time_array[i+1]) + time_units[i+1]
                return string
        return '0us0ns'

    def Memory(self, Bytes:int=0, KBytes:Union[float,int]=0, MBytes:Union[float,int]=0, GBytes:Union[float,int]=0, TBytes:Union[float,int]=0, precision=2):
        Total   = int(Bytes + KBytes*1024 + MBytes*1024*1024 + GBytes*1024*1024*1024 + TBytes*1024*1024*1024*1024)

        Bytes   = int(Total%1024)
        Total   = int(Total/1024)
        KBytes  = int(Total%1024)
        Total   = int(Total/1024)
        MBytes  = int(Total%1024)
        Total   = int(Total/1024)
        GBytes  = int(Total%1024)
        Total   = int(Total/1024)
        TBytes  = Total

        bytes_array = [int(TBytes), int(GBytes), int(MBytes), int(KBytes), int(Bytes)]
        bytes_units = ['TB', 'GB', 'MB', 'KB', 'B']
        max_idx = 0
        for max_idx, value in enumerate(bytes_array):
            if value != 0:
                break
        value = 0
        multiplier = 1
        for value_ in bytes_array[max_idx:]:
            value += value_/multiplier
            multiplier *= 1024
        value = value

        return str(round(value, precision)) + bytes_units[max_idx]

    def TextStyles(self):
        return self.Styles.copy()
    def Text(self, style, parameters={}):
        CurrentStyle = self.Styles[style].copy()
        CurrentStyle.update(parameters.items())
        return CurrentStyle

    def WrappedText(self, simbols_in_line, text):
        text_ = ''
        for i, l in enumerate(text):
            if (i+1) % simbols_in_line == 0:
                if l == ' ':
                    text_ += '\n'
                else:
                    for i in reversed(range(len(text_))):
                        if text_[i] == ' ':
                            text_ = text_[:i] + '\n' + text_[i:]
                            break
                    text_ += l
            else:
                text_ += l
        return text_

    def SmartWrappedText(self, text:str, max_symbols_in_line:int, separators:Union[List,Tuple]=(' ',)):
        String = ''

        while text != '':
            if len(text) <= max_symbols_in_line:
                String += text + '\n'
                text = ''
            else:
                string = text[:max_symbols_in_line]
                success = False
                for n, char in reversed(list(enumerate(string))):
                    if char in separators:
                        string = text[:n+1]
                        text = text[n+1:]
                        success = True
                        break
                if not success:
                    string = text[:max_symbols_in_line-1] + '-'
                    text = text[max_symbols_in_line-1:]
                String += string + '\n'

        if String[-1] == '\n':
            String = String[:-1]

        return String

Format = Formater()