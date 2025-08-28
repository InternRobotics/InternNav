import torch
import torch.nn as nn
import math
from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PositionalEncoding(nn.Module):
    """Positional encoding module"""
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]
    
class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding using nn.Embedding"""
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """
        x: Input token vectors with shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_encoding = self.position_embedding(position_ids)
        return position_encoding


class TokenCompressor(nn.Module):
    def __init__(self, embed_dim, num_heads, target_length):
        super(TokenCompressor, self).__init__()
        self.target_length = target_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable target sequence using nn.Embedding
        self.target_embedding = nn.Embedding(target_length, embed_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim) 
        
        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)

        # Multi-Head Attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, padding_mask=None):
        """
        x: (bs, N, 384) - Input sequence (variable length)
        padding_mask: (bs, N) - Padding mask for input sequence (True for padding positions)
        """
        bs, token_len, _ = x.shape
        
        # Add positional encoding to input
        token_pe = self.token_positional_encoding(x)
        x = x + token_pe
        
        query = self.target_embedding.weight.unsqueeze(0).expand(bs, -1, -1)

        # Get target sequence from embedding
        query_pe = self.query_positional_encoding(query)
        
        query = query + query_pe

        # Cross Attention: target is Query, x is Key and Value
        out, _ = self.cross_attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=padding_mask
        )
        return out

class DAT_RGBD_Patch_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 finetune=True,
                 memory_size=8,
                 checkpoint="checkpoints/depth_anything_v2_vits.pth",
                 input_dtype="bf16",
                 device = 'cuda:0'):
        super().__init__()
        self.finetune = finetune
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.input_dtype = torch.bfloat16 if input_dtype == "bf16" else torch.float32

        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.rgb_model = self.rgb_model.pretrained
        
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.input_dtype)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=self.input_dtype)

        if finetune:
            self.rgb_model.train()
        else:
            self.rgb_model.eval()

        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained
        self.depth_model.train()

        self.former_query = nn.Embedding(self.memory_size * 16, 384)
        nn.init.constant_(self.former_query.weight, val=0)
        self.former_pe = nn.Embedding((self.memory_size * 2) * 256, 384)
        nn.init.constant_(self.former_pe.weight, val=0)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images, depths):
        if len(images.shape) == 4:
            tensor_images = images.to(dtype=self.input_dtype).permute(0, 3, 1, 2)
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(images.device)) / self.preprocess_std.to(images.device).reshape(1, 3, 1, 1)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
        elif len(images.shape) == 5:
            B, T, H, W, C = images.shape
            tensor_images = images.to(dtype=self.input_dtype).permute(0, 1, 4, 2, 3)
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (tensor_images - self.preprocess_mean.to(images.device).reshape(1, 3, 1, 1)) / self.preprocess_std.to(images.device).reshape(1, 3, 1, 1)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B, T * 256, -1)

        if not self.finetune:
            image_token = image_token.detach()

        if len(depths.shape) == 4:
            tensor_depths = depths.to(dtype=self.input_dtype).permute(0, 3, 1, 2)
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(depths.shape) == 5:
            B, T, H, W, C = depths.shape
            tensor_depths = depths.to(dtype=self.input_dtype).permute(0, 1, 4, 2, 3)
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B, T * 256, -1)

        former_pe_indice = torch.arange((self.memory_size * 2) * 256, device=images.device).expand(image_token.shape[0], (self.memory_size * 2) * 256)
        former_pe = self.former_pe(former_pe_indice)
        former_token = torch.cat((image_token, depth_token), dim=1) + former_pe

        former_query_indice = torch.arange(self.memory_size * 16, device=images.device).expand(image_token.shape[0], self.memory_size * 16)
        former_query = self.former_query(former_query_indice)

        memory_token = self.former_net(former_query, former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 finetune=True,
                 memory_size=8,
                 checkpoint="checkpoints/depth_anything_v2_vits.pth",
                 device='cuda:0'):
        super().__init__()
        # ensure the device is valid
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.finetune = finetune
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        # TODO: Hack for navdp training using transformers 4.51.0 when loading the checkpoint
        self.rgb_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.rgb_model = self.rgb_model.pretrained.float()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
        if finetune:
            self.rgb_model.train()
        else:
            self.rgb_model.eval()
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256) 
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        self.to(device)
        
    def forward(self,images,depths):
        device = self._get_device()
        images = images.to(device)
        depths = depths.to(device)
        if len(images.shape) == 4:
            tensor_images = torch.as_tensor(images,dtype=torch.float32,device=device).permute(0,3,1,2)
            tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
            tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(device))/self.preprocess_std.reshape(1,3,1,1).to(device)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
        elif len(images.shape) == 5:
            tensor_images = torch.as_tensor(images,dtype=torch.float32,device=device).permute(0,1,4,2,3)
            B,T,C,H,W = tensor_images.shape
            tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
            tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(device))/self.preprocess_std.reshape(1,3,1,1).to(device)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
        if not self.finetune:
            image_token = image_token.detach()
        if len(depths.shape) == 4:
            tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=device).permute(0,3,1,2)
            tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
            tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(depths.shape) == 5:
            tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=device).permute(0,1,4,2,3)
            B,T,C,H,W = tensor_depths.shape
            tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
            tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
        former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
        former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384),device=device))
        memory_token = self.former_net(former_query,former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token
    def _get_device(self):
        """get device safely"""
        # try to get device through model parameters
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass
        
        # try to get device through buffer
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass
        
        # try to get device through submodule
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue
        
        # finally revert to default device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NavDP_ImageGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=6,
                                                            out_channels = self.imagegoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.imagegoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.imagegoal_encoder.patch_embed.proj.stride,
                                                            padding = self.imagegoal_encoder.patch_embed.proj.padding)
        self.imagegoal_encoder.train()
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images):
        assert len(images.shape) == 4 # B,C,H,W
        tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
        image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token

class NavDP_PixelGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=4,
                                                            out_channels = self.pixelgoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.pixelgoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.pixelgoal_encoder.patch_embed.proj.stride,
                                                            padding = self.pixelgoal_encoder.patch_embed.proj.padding)
        self.pixelgoal_encoder.train()
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images):
        assert len(images.shape) == 4 # B,C,H,W
        tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
        image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token