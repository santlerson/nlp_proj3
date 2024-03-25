from environments import environment
from utils.datasets import OfflineDataSet, OnlineSimulationDataSet, ConcatDatasets
from utils.samplers import NewUserBatchSampler, SimulationSampler
from consts import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from utils.usersvectors import UsersVectors
import torch.optim as optim
from itertools import chain
import wandb
from utils import *
import pickle
from utils import personas


class BetterEnvironment(environment.Environment):
    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
            # output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        # if update_vectors:
        #     self.currentDM = output["user_vector"]
        #     self.currentGame = output["game_vector"]
        return output


    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }


    def train(self, do_eval=True):
        print("Start training the environment...")
        online_sim_type = self.config["online_sim_type"]
        assert online_sim_type in ["None", "mixed", "before_epoch", "init"]
        phases = []

        if online_sim_type == "init":
            raise NotImplementedError("The 'init' simulation type is not implemented yet.")

        elif self.config["task"] == "on_policy":
            human_train_size = self.config["human_train_size"]
            test_size = ON_POLICY_TEST_SIZE
            real_users = np.random.choice(range(test_size, DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS), human_train_size,
                                          replace=False)
            llm_real_train = real_users[:human_train_size]
            llm_real_test = np.arange(test_size)

        if self.config["human_train_size"] != 0:
            # if self.config["human_train_size"] != -1 and self.config["human_train_size"] != "all":
            #     real_users = np.random.choice(range(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS), self.config["human_train_size"], replace=False)
            # else:
            #     real_users = np.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
            if self.config["ENV_HPT_mode"]:
                all_users = np.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
                train_users = np.random.choice(all_users,
                                               int(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS * 0.8), replace=False)
                test_users = np.setdiff1d(all_users, train_users)
                train_dataset = OfflineDataSet(user_groups="X", strategies=[3, 0, 2, 5], users=train_users,
                                               weight_type=self.config.loss_weight_type, config=self.config)
            else:
                train_dataset = OfflineDataSet(user_groups="X", weight_type=self.config.loss_weight_type,
                                               config=self.config)

            train_sampler = NewUserBatchSampler(train_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
            phases += [("Train", train_dataloader)]

        if self.config["offline_simulation_size"] != 0:
            if self.config.personas_group_number == -1:
                llm_users_options = range(TOTAL_LLM_USERS)
                llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]),
                                             replace=False)
            else:
                groups = personas.get_personas_in_group(self.config.personas_group_number)
                personas_df = pd.read_csv(self.config["OFFLINE_SIM_DATA_PATH"])
                if self.config["personas_balanced"]:
                    group_size = int(self.config["offline_simulation_size"]) // len(groups)
                    llm_users = []
                    for group in groups:
                        llm_users_options = personas_df[personas_df["persona"] == group]["user_id"].unique()
                        persona_users = np.random.choice(llm_users_options, group_size, replace=False)
                        llm_users += [persona_users]
                    llm_users = np.concatenate(llm_users)
                else:
                    llm_users_options = personas_df[personas_df["persona"].isin(groups)]["user_id"].unique()
                    llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]),
                                                 replace=False)
            offline_dataset = OfflineDataSet(user_groups="L", users=llm_users, config=self.config,
                                             weight_type=self.config.loss_weight_type,
                                             strategies=self.config.strategies)
            offline_sim_sampler = NewUserBatchSampler(offline_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            offline_sim_dataloader = DataLoader(offline_dataset, batch_sampler=offline_sim_sampler, shuffle=False)
            phases.insert(0, ("Offline Simulation", offline_sim_dataloader))

        if do_eval:
            if self.config["ENV_HPT_mode"]:
                test_dataset = OfflineDataSet(user_groups="X", users=test_users, strategies=[19, 59],
                                              weight_type=self.config.loss_weight_type, config=self.config)
            elif self.config["task"] == "off_policy":
                test_dataset = OfflineDataSet(user_groups="Y", strategies=self.config.strategies,
                                              weight_type="sender_receiver", config=self.config)
            else:
                assert self.config["task"] == "on_policy"
                test_dataset = OfflineDataSet(user_groups="X", users=llm_real_test, weight_type="sender_receiver",
                                              config=self.config,
                                              strategies=self.config.strategies)

            test_sampler = NewUserBatchSampler(test_dataset, batch_size=ENV_BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, shuffle=False)
            phases += [("Test", test_dataloader)]

        if self.config["online_simulation_size"] > 0 and online_sim_type == "before_epoch":
            phases.insert(0, ("Online Simulation", "sim_dataloader"))

        self.model.to(device)
        optimizer = torch.optim.Adam([p for p in chain(self.model.parameters()) if p.requires_grad],
                                     lr=self.env_learning_rate, weight_decay=self.config["weight_decay"])
        self.set_train_mode()
        metrics = Metrics("ENV")
        for epoch in range(self.config["total_epochs"]):
            metrics.write("epoch", epoch)
            result_saver = {phase:  ResultSaver(config=self.config, epoch=epoch) for phase, _ in phases}
            print("#" * 16)
            print(f"# Epoch {epoch}")
            print("#" * 16)
            if self.config["online_simulation_size"] > 0 and online_sim_type in ["before_epoch", "mixed"]:
                online_simulation_dataset = OnlineSimulationDataSet(config=self.config)
                online_simulation_sampler = SimulationSampler(online_simulation_dataset, SIMULATION_BATCH_SIZE)
                online_simulation_dataloader = DataLoader(online_simulation_dataset,
                                                          batch_sampler=online_simulation_sampler, shuffle=False)
            for phase, dataloader in phases:
                # print(phase)
                metrics.set_stage(phase)
                if phase == "Online Simulation" and online_sim_type == "before_epoch":
                    dataloader = online_simulation_dataloader
                if self.use_user_vector:
                    self.model.user_vectors.delete_all_users()
                    self.model.game_vectors.delete_all_users()
                total_loss = 0
                total_weight = 0
                total_proba_to_right_action = 0
                total_proba_weighted = 0
                total_right_action = 0
                total_right_action_weighted = 0
                total_weight = 0
                n_actions = 0
                if phase != "Test":
                    self.set_train_mode()
                    if online_sim_type == "mixed":
                        dataloader = ConcatDatasets(dataloader, online_simulation_dataloader)
                else:
                    self.set_eval_mode()
                for batch in tqdm(dataloader, desc=phase):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    batch_size, _ = batch["hotels_scores"].shape
                    review_vector = batch["review_vector"].reshape(batch_size, DATA_ROUNDS_PER_GAME, -1)
                    round_end_vars = [batch[feature].unsqueeze(-1) for feature in END_ROUND_FEATURES]
                    round_end_vars = torch.cat(round_end_vars, dim=2).double().to(device)
                    review_vector = review_vector.double().to(device)
                    model_vectors = {
                                     "begin_round": review_vector,
                                     "end_round": round_end_vars,
                                     "labels": batch["action_taken"].to(device)
                                    }

                    if self.use_user_vector:
                        model_vectors["user_vector"] = self.model.user_vectors[batch["user_id"].to("cpu").numpy()].to(
                            device)
                        model_vectors["game_vector"] = self.model.game_vectors[batch["user_id"].to("cpu").numpy()].to(
                            device)
                    padding_anti_mask = batch["action_taken"] != -100
                    if phase != "Test":
                        model_output = self.model(model_vectors, padding_mask=~padding_anti_mask)
                    else:
                        with torch.no_grad():
                            model_output = self.model(model_vectors, padding_mask=~padding_anti_mask)
                    output = model_output["output"]
                    mask = (padding_anti_mask).flatten()
                    relevant_predictions = output.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1)[mask]
                    relevant_ground_truth = batch["action_taken"].flatten()[mask]
                    relevant_weight = batch["weight"][batch["is_sample"]]

                    proba_to_right_action = torch.exp(
                        relevant_predictions[torch.arange(len(relevant_predictions), device=device),
                                             relevant_ground_truth])
                    total_proba_to_right_action += proba_to_right_action.sum().item()
                    total_proba_weighted += (proba_to_right_action * relevant_weight).sum().item()
                    total_right_action += (proba_to_right_action >= 0.5).sum().item()
                    total_right_action_weighted += ((proba_to_right_action >= 0.5) * relevant_weight).sum().item()
                    n_actions += len(proba_to_right_action)
                    target = batch["action_taken"].reshape(-1)[batch["is_sample"].reshape(-1)]
                    total_weight += batch["weight"][batch["is_sample"]].sum().item()
                    losses = (self.loss_fn(relevant_predictions, relevant_ground_truth) * relevant_weight
                            )
                    loss=losses.mean()
                    total_loss += losses.sum().item()
                    total_weight += relevant_weight.sum().item()
                    if phase != "Test":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    result_saver[phase].add_results(ids=batch["action_id"].flatten()[mask].cpu(),
                                                 user_id=torch.repeat_interleave(batch["user_id"],
                                                                                 batch["bot_strategy"].shape[-1])[
                                                     mask].cpu(),
                                                 bot_strategy=batch["bot_strategy"].flatten()[mask].cpu(),
                                                 accuracy=proba_to_right_action.cpu())

                    if self.use_user_vector:
                        updated_user_vectors = model_output["user_vector"].to("cpu").detach()
                        self.model.user_vectors[batch["user_id"].to("cpu").numpy()] = updated_user_vectors.squeeze()
                        updated_game_vectors = model_output["game_vector"].to("cpu").detach()
                        self.model.game_vectors[batch["user_id"].to("cpu").numpy()] = updated_game_vectors.squeeze()

                metrics.write("TotalLoss", total_loss)
                metrics.write("AverageLoss", total_loss/total_weight if total_weight else 0)
                if n_actions:
                    metrics.write("Right action", total_right_action / n_actions)
                    metrics.write("Probability to choose the right action", total_proba_to_right_action / n_actions)
                if total_weight:
                    metrics.write("Weighted right action", total_right_action_weighted / total_weight)
                    metrics.write("Weighted probability to choose the right action:",
                                  total_proba_weighted / total_weight)

                results_df = result_saver[phase].next_epoch()
                for prefix in ["proba_", ""]:
                    prefix=phase+"_"+prefix
                    if "proba_" not in prefix:
                        results_df["Accuracy"] = results_df["Accuracy"] > 0.5
                    accuracy = results_df["Accuracy"].mean()
                    metrics.write(prefix + "accuracy", accuracy)
                    for strategy in results_df["Bot_Strategy"].unique():
                        bot_accuracy = results_df[results_df["Bot_Strategy"] == strategy]["Accuracy"].mean()
                        metrics.write(prefix + f"accuracy_strategy_{strategy}", bot_accuracy)
                    accuracy_per_mean_strategy = results_df.groupby("Bot_Strategy").mean()["Accuracy"].mean()
                    metrics.write(prefix + f"accuracy_per_mean_strategy", accuracy_per_mean_strategy)
                    accuracy_per_mean_user = results_df.groupby("User_ID").mean()["Accuracy"].mean()
                    metrics.write(prefix + "accuracy_per_mean_user", accuracy_per_mean_user)
                    accuracy_per_mean_user_and_bot = results_df.groupby(["User_ID", "Bot_Strategy"]).mean()[
                        "Accuracy"].mean()
                    metrics.write(prefix + "accuracy_per_mean_user_and_bot", accuracy_per_mean_user_and_bot)
                    print(prefix + "accuracy_per_mean_user_and_bot: ", accuracy_per_mean_user_and_bot)
                wandb.log(metrics.all)
            metrics.next_epoch()
        self.model.to("cpu")
        self.set_eval_mode()
