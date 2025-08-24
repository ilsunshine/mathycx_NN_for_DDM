import subprocess
import sys


def run_commands(commands):
    for cmd in commands:
        print(f"执行命令{cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {cmd}")
            sys.exit(1)


if __name__ == "__main__":
    # 用于生成并依次执行命令脚本
    py_dir = "./model_for_eigvalue.py"
    # args={
    #     "Experimental_args": ["scheduler.MultiStepLR.milestones","weight_network.AdaptivePoolEncoder.feature_dim","weight_network.AdaptivePoolEncoder.output_dim","training.epochs"],
    #     "number_for_avg":5,#每个参数设置运行的次数，用于测试平均性能
    #     "weight_network.AdaptivePoolEncoder.feature_dim":[16,16,8,8],
    #     "weight_network.AdaptivePoolEncoder.output_dim": [256, 512, 256,512],
    #     "scheduler.MultiStepLR.milestones":[[10,20,30],[10,20,30],[10,20],[10,20]],
    #     "training.epochs":[40,40,40,40]
    # }
    args = {
        "Experimental_args": ["preditor_network.update_c", "subdomain_network.FCResNet_Block.layer_size", ],
        "number_for_avg": 1,  # 每个参数设置运行的次数，用于测试平均性能
        # "preditor_network.MLP.act_fun":["relu","tanh","relu","tanh","relu","tanh"],
        # "weight_network.AdaptivePoolEncoder.output_dim": [256, 512, 256,512],
        # "scheduler.MultiStepLR.milestones":[[10,20,30],[10,20,30],[10,20],[10,20]],
        "preditor_network.update_c": [0.5, 0.8, 0.95],
        # "model.model_network":["Step_function_network_with_omegann","Step_function_network_with_omegann","Step_function_network_with_omegann","Step_function_network_with_omegann","Step_function_network_with_omegann","Step_function_network_with_omegann"],
        "subdomain_network.FCResNet_Block.layer_size": [[16, 16, 32, 32], [16, 16, 32, 32], [16, 16, 32, 32]],
        # "subdomain_network.FCResNet_Block.layer_size":[[16,16,16,16],[16,16,16,16],[16,16,16,16]],
        # "subdomain_network.FCResNet_Block.layer_size":[[32,32,16,16],[32,32,16,16],[32,32,16,16]],
        # "subdomain_network.FCResNet_Block.layer_size":[[8,16,32,32],[8,16,32,32],[8,16,32,32]],
        # "subdomain_network.FCResNet_Block.if_batchnorm": ["False",False,False],
        # "subdomain_network.FCResNet_Block.if_batchnorm": [True,True,True,True,True,True],

        # "preditor_network.initial_c": [0.2,0.2,0.2,5.0,5.0,5.0],
        # "preditor_network.update_c": [0.5,0.65,0.8,0.5,0.65,0.8]
    }
    commands = []
    experiment_len = len(args[args["Experimental_args"][0]])
    for i in range(experiment_len):
        for j in range(args["number_for_avg"]):
            command = "python " + py_dir + " "
            for arg_name in args["Experimental_args"]:
                arg_value = args[arg_name][i]
                if type(args[arg_name][i]) == list:
                    command += "--" + arg_name + '=' + "\"" + str(arg_value) + "\" "
                elif type(args[arg_name][i]) == bool:
                    command += "--" + arg_name + ' ' + "\"" + str(arg_value) + "\" "
                else:
                    command += "--" + arg_name + '=' + str(arg_value) + " "
            commands.append(command)
    commands.append("sleep 800s")  # 等待时间确保其他进程已经完成
    commands.append("shutdown")
    run_commands(commands)
    print("所有命令执行完成")