from typing import Union, List, Iterable, Any, Tuple
import sys
from Formaters import Format
from time import time as timer
import numpy
class CycleTimePredictor:
    DataFunctions : List
    IteratingList : Union[List, Iterable, Any]
    TotalIterations : int
    CurrentIteration : int
    StartTime : float
    StringLength : int

    FullTimePredictions : numpy.ndarray
    FullTime : float
    def __init__(self, IteratingList:Union[List, Iterable], AdditionalDataFunctions:Union[List,Tuple,Iterable]=None):
        if AdditionalDataFunctions is None:
            AdditionalDataFunctions = []
        if type(AdditionalDataFunctions) is Tuple or type(AdditionalDataFunctions) is Iterable:
            AdditionalDataFunctions = list(AdditionalDataFunctions)
        self.IteratingList = IteratingList
        self.DataFunctions = AdditionalDataFunctions
        self.TotalIterations = len(IteratingList)
        self.CurrentIteration = 0
        self.StartTime = timer()
        self.StringLength = 0
        self.FullTimePredictions = numpy.zeros(len(IteratingList))
    def __iter__(self):
        iter(self.IteratingList)
        return self
    def __next__(self):
        self.CurrentIteration += 1
        CurrentTime = timer()
        TimePassed = CurrentTime-self.StartTime
        TimePerIteration = TimePassed/self.CurrentIteration
        TimeLeft = TimePerIteration * (self.TotalIterations-self.CurrentIteration+1)
        TimeTotal = TimePerIteration * (self.TotalIterations + 1)

        string = 'Итерация: ' + str(self.CurrentIteration-1) + ' из ' + str(self.TotalIterations)
        string += ' | Прошло времени: ' + Format.Time(TimePassed) + ' из ' + Format.Time(TimeTotal) + ', Осталось: ' + Format.Time(TimeLeft)
        for function in self.DataFunctions:
            string += ' | ' + function()
        StringLength = len(string)
        string = '\033[35m{}\033[0m'.format(string)
        sys.stdout.write(f"\r{string + ' '*(self.StringLength - StringLength)}")
        self.StringLength = StringLength

        if self.CurrentIteration-1 == self.TotalIterations:
            self.FullTime = TimePassed

            self.FullTimePredictions = self.FullTimePredictions[1:]
            deviation = numpy.mean(numpy.abs(self.FullTimePredictions-self.FullTime))
            # print('\n\t|| Среднее отклонения предсказания времени: ' + Format.Time(deviation))
            print('')
            raise StopIteration
        self.FullTimePredictions[self.CurrentIteration - 1] = TimeTotal

        self.IteratingList = iter(self.IteratingList)
        return next(self.IteratingList)
    def __len__(self):
        return len(self.IteratingList)
