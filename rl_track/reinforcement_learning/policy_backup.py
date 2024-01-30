import pufferlib
import pufferlib.emulation
import pufferlib.models
import torch
import torch.nn.functional as F
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class Random(pufferlib.models.Policy):
    """A random policy that resets weights on every call"""

    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(1, n) for n in envs.single_action_space.nvec]
        )

    def encode_observations(self, env_outputs):
        return torch.randn((env_outputs.shape[0], 1)).to(env_outputs.device), None

    def decode_actions(self, hidden, lookup):
        torch.nn.init.xavier_uniform_(hidden)
        actions = [dec(hidden) for dec in self.decoders]
        return actions, None

    def critic(self, hidden):
        return torch.zeros((hidden.shape[0], 1)).to(hidden.device)


class Baseline(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        # self.proj_fc = torch.nn.Linear(5 * input_size, input_size)
        self.proj_fc = torch.nn.Linear(1536, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

        self.mean_pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

    def encode_observations(self, flat_observations):
        """
        player_embeddings、item_embedding做pooling
        market_embeddings做mean_pooling
        mean pooling和max pooling
        """
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations,
            self.flat_observation_space,
            self.flat_observation_structure,
        )

        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )

        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        inventory = self.inventory_encoder(item_embeddings)  # fc

        market_embeddings = self.item_encoder(env_outputs["Market"])  # no_pooling
        market = self.market_encoder(
            market_embeddings
        )  # fc +mean pooling already applied

        task = self.task_encoder(env_outputs["Task"])

        # print(f"emb player:{player_embeddings.shape}, item:{item_embeddings.shape}, market_embeddings:{market_embeddings.shape}")
        # print(f"tile:{tile.shape}, my_agent:{my_agent.shape}, inventory:{inventory.shape}, market:{market.shape}")

        # pooled_player_embeddings = self.mean_pool(player_embeddings)
        # pooled_item_embeddings = self.mean_pool(item_embeddings)
        pooled_item_embeddings = item_embeddings.mean(dim=1)
        pooled_player_embeddings = player_embeddings.mean(dim=1)

        print(
            f"tile:{tile.shape}, my_agent:{my_agent.shape},pooled_agent:{pooled_player_embeddings.shape},pooled_item:{pooled_item_embeddings.shape},pooled_market:{market.shape}, task.shape:{task.shape}"
        )
        # obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1) # original

        obs = torch.cat(
            [
                tile,
                my_agent,
                pooled_player_embeddings,
                pooled_item_embeddings,
                market,
                task,
            ],
            dim=-1,
        )
        obs = self.proj_fc(obs)

        embeddings = [player_embeddings, item_embeddings, market_embeddings]
        padded_embeddings = []
        for embedding in embeddings:
            padding_size = 1  # The size of padding to be added
            padding = torch.zeros(
                embedding.size(0),
                padding_size,
                embedding.size(2),
                device=embedding.device,
            )
            padded_embedding = torch.cat([embedding, padding], dim=1)
            padded_embeddings.append(padded_embedding)
        # Replace the original embeddings with the padded versions
        player_embeddings, item_embeddings, market_embeddings = padded_embeddings

        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class TileEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tile_offset = torch.tensor([i * 256 for i in range(3)])
        self.embedding = torch.nn.Embedding(3 * 256, 32)

        self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    def forward(self, tile):
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7
        tile = self.embedding(
            tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
        )

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_fc(tile))

        return tile


class PlayerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.entity_dim = 31
        """
     [
    "id",
    "npc_type", # 1 - passive, 2 - neutral, 3 - aggressive
    "row",
    "col",

    # Status
    "damage",
    "time_alive",
    "freeze",
    "item_level",
    "attacker_id",
    "latest_combat_tick",
    "message",

    # Resources
    "gold",
    "health",
    "food",
    "water",

    # Combat Skills
    "melee_level",
    "melee_exp",
    "range_level",
    "range_exp",
    "mage_level",
    "mage_exp",

    # Harvest Skills
    "fishing_level",
    "fishing_exp",
    "herbalism_level",
    "herbalism_exp",
    "prospecting_level",
    "prospecting_exp",
    "carving_level",
    "carving_exp",
    "alchemy_level",
    "alchemy_exp",
  ]
    """
        self.num_classes_npc_type = 3  # only using npc_type for one hot

        # self.player_offset = torch.tensor([i * 256 for i in range(self.entity_dim)])
        # self.embedding = torch.nn.Embedding((self.entity_dim+self.num_classes_npc_type-1) * 256, 32)
        # self.agent_fc = torch.nn.Linear(self.entity_dim * 32, hidden_size)
        # self.my_agent_fc = torch.nn.Linear(self.entity_dim * 32, input_size)

        self.agent_fc = torch.nn.Linear(self.entity_dim + 3 - 1, hidden_size)
        self.my_agent_fc = torch.nn.Linear(self.entity_dim + 3 - 1, input_size)

    def print_device_info(self):
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Device: {param.device}")

    def forward(self, agents, my_id):
        npc_type = agents[:, :, 1]
        one_hot_npc_type = F.one_hot(
            npc_type.long(), num_classes=self.num_classes_npc_type
        ).float()  # Subtract 1 if npc_type starts from 1
        one_hot_agents = torch.cat(
            [agents[:, :, :1], one_hot_npc_type, agents[:, :, 2:]], dim=-1
        )
        # print(f"original_agents_shape:{agents.shape}, one_hot_agents_shape:{one_hot_agents.shape}")

        # agent_embeddings = self.embedding(
        #     agents.long().clip(0, 255) + self.player_offset.to(agents.device)
        # )
        # agent_embeddings = self.embedding(
        #     one_hot_agents.long().clip(0, 255).to(one_hot_agents.device)
        # )

        agent_ids = one_hot_agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        # batch, agent, attrs, embed = agent_embeddings.shape

        # # Embed each feature separately
        # agent_embeddings = agent_embeddings.view(batch, agent, attrs * embed)

        my_agent_embeddings = one_hot_agents[
            torch.arange(one_hot_agents.shape[0]), row_indices
        ]
        # print("agent_embeddings:",agent_embeddings.shape)
        # self.print_device_info()
        # print(f"my_id:{my_id},agent_embeddings:{agent_embeddings.shape}, my_agent_embeddings:{my_agent_embeddings.shape}")
        agent_embeddings = self.agent_fc(one_hot_agents.cuda())
        my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
        my_agent_embeddings = F.relu(my_agent_embeddings)

        return agent_embeddings, my_agent_embeddings


class ItemEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # self.item_offset = torch.tensor([i * 256 for i in range(16)])
        # self.embedding = torch.nn.Embedding(256, 32)
        # self.fc = torch.nn.Linear(2 * 32 + 12, hidden_size)

        self.fc = torch.nn.Linear(17 + 2 + 14 - 2, hidden_size)
        self.discrete_idxs = [1, 14]  # type_id:0,1 ...16= 17", equipped:0,1=2"
        self.discrete_offset = torch.Tensor([2, 0])
        self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        self.continuous_scale = torch.Tensor(
            [
                1 / 10,
                1 / 10,
                1 / 10,
                1 / 100,
                1 / 100,
                1 / 100,
                1 / 40,
                1 / 40,
                1 / 40,
                1 / 100,
                1 / 100,
                1 / 100,
            ]
        )

    def print_device_info(self):
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Device: {param.device}")

    def forward(self, items):
        self.print_device_info()
        if self.discrete_offset.device != items.device:
            self.discrete_offset = self.discrete_offset.to(items.device)
            self.continuous_scale = self.continuous_scale.to(items.device)

        # discrete = items[:, :, self.discrete_idxs] + self.discrete_offset
        # discretes = items[:, :, self.discrete_idxs]

        one_hot_discrete_equipped = F.one_hot(
            items[:, :, 1].long(), num_classes=2
        ).float()
        one_hot_discrete_type_id = F.one_hot(
            items[:, :, 14].long(), num_classes=17
        ).float()
        one_hot_discrete = torch.concat(
            [one_hot_discrete_type_id, one_hot_discrete_equipped], dim=-1
        )  #
        # print("Item:one_hot_discrete_shape:",one_hot_discrete.shape)

        # Embed each feature separately
        # discrete = self.embedding(one_hot_discrete.long().clip(0, 255))
        # batch, item, attrs, embed = discrete.shape
        # discrete = discrete.view(batch, item, attrs * embed)

        continuous = items[:, :, self.continuous_idxs] / self.continuous_scale

        item_embeddings = torch.cat([one_hot_discrete, continuous], dim=-1)
        print("bf_fc_item_embeddings.shape:", item_embeddings.shape)
        item_embeddings = self.fc(item_embeddings)
        # print("item_embeddings.shape:",item_embeddings.shape)
        return item_embeddings


class InventoryEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(12 * hidden_size, input_size)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
        return self.fc(inventory).mean(1)


class MarketEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, market):
        h = self.fc(market)
        pooled_market_embeddings = h.mean(dim=1)

        print(
            f"market_embedding.shape:{h.shape}, pooled_market_emd.shpae:{pooled_market_embeddings.shape}"
        )

        return pooled_market_embeddings


class TaskEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, task_size):
        super().__init__()
        self.fc = torch.nn.Linear(task_size, input_size)

    def forward(self, task):
        return self.fc(task.clone())


class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "attack_style": torch.nn.Linear(hidden_size, 3),
                "attack_target": torch.nn.Linear(hidden_size, hidden_size),
                "market_buy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_destroy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_item": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_player": torch.nn.Linear(hidden_size, hidden_size),
                "gold_quantity": torch.nn.Linear(hidden_size, 99),
                "gold_target": torch.nn.Linear(hidden_size, hidden_size),
                "move": torch.nn.Linear(hidden_size, 5),
                "inventory_sell": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_price": torch.nn.Linear(hidden_size, 99),
                "inventory_use": torch.nn.Linear(hidden_size, hidden_size),
            }
        )

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            inventory_embeddings,
            market_embeddings,
            action_targets,
        ) = lookup

        embeddings = {
            "attack_target": player_embeddings,
            "market_buy": market_embeddings,
            "inventory_destroy": inventory_embeddings,
            "inventory_give_item": inventory_embeddings,
            "inventory_give_player": player_embeddings,
            "gold_target": player_embeddings,
            "inventory_sell": inventory_embeddings,
            "inventory_use": inventory_embeddings,
        }

        action_targets = {
            "attack_style": action_targets["Attack"]["Style"],
            "attack_target": action_targets["Attack"]["Target"],
            "market_buy": action_targets["Buy"]["MarketItem"],
            "inventory_destroy": action_targets["Destroy"]["InventoryItem"],
            "inventory_give_item": action_targets["Give"]["InventoryItem"],
            "inventory_give_player": action_targets["Give"]["Target"],
            "gold_quantity": action_targets["GiveGold"]["Price"],
            "gold_target": action_targets["GiveGold"]["Target"],
            "move": action_targets["Move"]["Direction"],
            "inventory_sell": action_targets["Sell"]["InventoryItem"],
            "inventory_price": action_targets["Sell"]["Price"],
            "inventory_use": action_targets["Use"]["InventoryItem"],
        }

        actions = []
        for key, layer in self.layers.items():
            mask = None
            mask = action_targets[key]
            embs = embeddings.get(key)
            if embs is not None and embs.shape[1] != mask.shape[1]:
                print(f"key:{key}, embs shape:{embs.shape}, mask shape:{mask.shape}")
                b, _, f = embs.shape
                zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
                embs = torch.cat([embs, zeros], dim=1)

            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)

        return actions
