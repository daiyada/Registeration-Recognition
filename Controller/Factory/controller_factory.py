"""
@file controller_factory.py
@brief コントローラクラスのオブジェクト化を制御

@author Shunsuke Hishida / created on 2022/10/11
@copyrights(c) 2022 Global Walkers,inc All rights reserverd.
"""
import os
import sys

sys.path.append("../../")
from Controller.controller import IdentificationController, RegistrationController

class ControllerFactory(object):
    def __init__(self, executable_name: str) -> None:
        """Constructor

        Args:
            executable_name (str): 実行ファイル名 (拡張子あり)
            ex:
                execute_registration_on_local.py
        """
        _, operation_name, _, place = (os.path.splitext(executable_name)[0]).split("_")
        ope_cls_name = operation_name.capitalize() + "Controller"
        obj = eval(ope_cls_name)()
        func_name = "predict_on_" + place
        predict_on_local = getattr(obj, func_name)
        predict_on_local()

