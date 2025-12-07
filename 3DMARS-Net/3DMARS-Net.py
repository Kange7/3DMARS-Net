from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import tf_util
import time
from tool import DataProcessing as DP
import pandas as pd


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d', time.gmtime())
                self.saving_path = self.saving_path + '_' + dataset.name
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        # This function is for variable sharing, including variables from tf.get_variable() and tf.Variable()
        with tf.variable_scope('inputs'):
            # dict() creates a dictionary.
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['dropout'] = flat_inputs[4 * num_layers + 4]    # Add dropout to prevent overfitting

            self.inputs['col'] = flat_inputs[4 * num_layers + 5: 5 * num_layers + 5]   # Color information, enhances representational ability
            self.inputs['high'] = flat_inputs[5 * num_layers + 5: 6 * num_layers + 5]  # High-level neighbor indices for cross-layer connections or neighbor enhancement


            self.labels = self.inputs['labels']
            # This can be understood as a formal parameter, defined for the process and assigned values during execution.
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.loss_type = 'sqrt'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training) 

        with tf.variable_scope('loss'):  # Define the loss calculation process, but removed the original ignore


            self.logits = tf.reshape(self.logits, [-1, self.config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])
            self.dropout_idx = tf.reshape(self.inputs['dropout'], [-1])

            valid_logits = self.logits

            valid_labels = self.labels

            self.loss = self.get_loss_improved(valid_logits, valid_labels, self.class_weights)  


        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=1)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def apply_ropev1(self, q, k, rel_xyz, max_freq=10.0):
        # rel_xyz: (B, N, K, 3)
        # Use sin/cos to construct frequency encoding
        pos = rel_xyz  # Can add scaling and nonlinear mapping
        # freq_seq = tf.linspace(1.0, max_freq, q.shape[-1] // 6)  # Different encoding frequency per dimension
        dim_q = tf.shape(q)[-1]
        freq_seq = tf.linspace(1.0, max_freq, dim_q // 6)  
        # freq_seq = tf.linspace(1.0, max_freq, num_freqs)
        freq_seq = tf.reshape(freq_seq, [1, 1, 1, -1])  # broadcast
        angles = pos[..., :1] * freq_seq  # Take x-axis as example, can extend to xyz
        sin_enc = tf.sin(angles)
        cos_enc = tf.cos(angles)

        # Split q/k and apply rotation
        q1, q2 = tf.split(q, 2, axis=-1)
        k1, k2 = tf.split(k, 2, axis=-1)
        q_rot = tf.concat([q1 * cos_enc - q2 * sin_enc, q1 * sin_enc + q2 * cos_enc], axis=-1)
        k_rot = tf.concat([k1 * cos_enc - k2 * sin_enc, k1 * sin_enc + k2 * cos_enc], axis=-1)
        return q_rot, k_rot
    
    def apply_rope(self, x, pos, max_freq=10.0):
        """
        x: [..., C] Tensor
        pos: [..., D] position tensor, like relative_xyz
        RoPE is only applied on the first 2*D dimensions
        """
        orig_shape = tf.shape(x)
        C = x.shape[-1]
        D = pos.shape[-1]

        # Construct rotation frequency (log scale)
        freq_seq = tf.pow(10000.0, tf.cast(tf.range(0, C // (2 * D)), tf.float32) / tf.cast(C // D, tf.float32))  # shape [C//(2*D)]
        freq_seq = tf.reshape(freq_seq, [1, 1, 1, -1])  # for broadcasting

        pos = tf.expand_dims(pos, axis=-1)  # [..., D, 1]
        angles = pos * freq_seq  # [..., D, F]
        sin_enc = tf.sin(angles)
        cos_enc = tf.cos(angles)

        sin_enc = tf.reshape(sin_enc, tf.concat([tf.shape(pos)[:-2], [-1]], axis=0))  # [..., D*F]
        cos_enc = tf.reshape(cos_enc, tf.concat([tf.shape(pos)[:-2], [-1]], axis=0))  # [..., D*F]

        x1, x2 = tf.split(x, 2, axis=-1)
        x_rot = tf.concat([x1 * cos_enc - x2 * sin_enc, x1 * sin_enc + x2 * cos_enc], axis=-1)
        return x_rot
    
    def apply_ropev3(self, x, relative_pos):
        """
        x: (B, N, K, C) - must have even dimension
        relative_pos: (B, N, K, 3)
        """

        with tf.variable_scope('rope', reuse=tf.AUTO_REUSE):
            dim = x.get_shape().as_list()[-1]
            if dim is None:
                dim = tf.shape(x)[-1]

            # Ensure last dimension of x is even
            assert_op = tf.assert_equal(dim % 2, 0, message='x dim must be even for rotary embedding')
            with tf.control_dependencies([assert_op]):
                x1, x2 = tf.split(x, 2, axis=-1)

            # Construct sin, cos encoding: this section can be adjusted based on your RoPE design
            # This is just a demonstration; actual implementation may require a more complex position encoding function
            freq = tf.exp(-tf.range(0., dim//2, 1.0) / (dim//2))
            freq = tf.reshape(freq, [1, 1, 1, -1])  # (1, 1, 1, d/2)

            # Project xyz (assuming 3D), get rotation angle (example uses only one component)
            theta = relative_pos[..., 0:1]  # (B, N, K, 1)

            cos_enc = tf.cos(theta * freq)  # (B, N, K, d/2)
            sin_enc = tf.sin(theta * freq)

            x_rot = tf.concat([
                x1 * cos_enc - x2 * sin_enc,
                x1 * sin_enc + x2 * cos_enc
            ], axis=-1)

            return x_rot
    
    def apply_ropev4(self, x, relative_pos):
        """
        x: (B, N, K, C) - must have even dimension
        relative_pos: (B, N, K, 3)
        """
        with tf.variable_scope('rope', reuse=tf.AUTO_REUSE):
            dim = x.get_shape().as_list()[-1]
            if dim is None:
                dim = tf.shape(x)[-1]

            # Force even dimension
            assert_op = tf.assert_equal(dim % 2, 0, message='x dim must be even for rotary embedding')
            with tf.control_dependencies([assert_op]):
                x1, x2 = tf.split(x, 2, axis=-1)

            # Construct frequency encoding, range similar to original Transformer settings
            inv_freq = 1.0 / tf.pow(10000.0, tf.cast(tf.range(0, dim // 2, 1), tf.float32) / (dim // 2))
            inv_freq = tf.reshape(inv_freq, [1, 1, 1, -1])  # (1, 1, 1, d/2)

            # Use all coordinate axes, combine into rotation angle: use norm or arbitrary mapping
            theta = tf.norm(relative_pos, axis=-1, keepdims=True)  # (B, N, K, 1)

            # Apply RoPE
            sinusoid_inp = theta * inv_freq  # (B, N, K, d/2)
            cos_enc = tf.cos(sinusoid_inp)
            sin_enc = tf.sin(sinusoid_inp)

            x_rot = tf.concat([
                x1 * cos_enc - x2 * sin_enc,
                x1 * sin_enc + x2 * cos_enc
            ], axis=-1)

            return x_rot

    def mult_latent_att(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):  # My improvement 1
        d_latent = d_out // num_head    # Minimum is 2

        batch_size = tf.shape(feature)[0]
        num_points = tf.shape(feature)[1]
        num_neigh = tf.shape(neigh)[-1]

        # 1. Position encoding
        xyz_center = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num_neigh, 1])
        relative_xyz = xyz_center - xyz  # (B, N, K, 3)
        pos_enc = tf.layers.dense(relative_xyz, d_out, name=name + '_xyz_enc')
        pos_enc = self.GLUE(tf.layers.batch_normalization(pos_enc, -1, 0.99, 1e-6, training=is_training))

        # 2. Feature high-dimensional projection
        feature = self.mlp(feature, d_out, name + '_feat_proj', is_training)
        neigh_feat = self.random_gather(feature, neigh)  # (B, N, K, C)
        feat_enc = tf.concat([neigh_feat, pos_enc], axis=-1)  # (B, N, K, 2C)
        feat_enc = tf.layers.dense(feat_enc, d_out, activation=tf.nn.leaky_relu, name=name + '_fused')

        # 3. Learnable latent vector queries initialization (shared across batch) , used dynamic tensor, but need static (still issue with d_latent grouping)
        latent_queries = tf.get_variable(name + '_latent_queries',
                                        shape=[num_head, 1, 1, d_latent],  # (H, 1, 1, C/H)  <== Problematic shape here
                                        initializer=tf.glorot_uniform_initializer())
        latent_queries = tf.tile(latent_queries, [1, batch_size, num_points, 1])  # (H, B, N, C/H)

        # 4. Attention map & aggregation
        outputs = []
        for h in range(num_head):
            q = latent_queries[h]  # (B, N, C/H)
            q = tf.squeeze(q, axis=0) if len(q.shape) == 4 else q
            k = tf.layers.dense(feat_enc, d_latent, name=f'{name}_head{h}_k')  # (B, N, K, C/H)
            v = tf.layers.dense(feat_enc, d_latent, name=f'{name}_head{h}_v')
            v = self.apply_ropev4(v, relative_xyz)

            attn = tf.nn.softmax(tf.einsum('bnc,bnkc->bnk', q, k) / tf.math.sqrt(tf.cast(d_latent, tf.float32)), axis=-1)  # (B, N, K)
            attn = tf.expand_dims(attn, -1)
            weighted_v = tf.reduce_sum(attn * v, axis=2)  # (B, N, C/H)
            outputs.append(weighted_v)

        # 5. Merge multi-head outputs
        out = tf.concat(outputs, axis=-1)  # (B, N, C)

        out.set_shape([None, None, d_out])  # Force tell TF: last dimension is d_out

        out = tf.layers.dense(out, d_out, name=name + '_final_proj')  # Last dimension of out is 0, cannot execute this statement
        out = self.GLUE(tf.layers.batch_normalization(out, -1, 0.99, 1e-6, training=is_training))
        return out
    
    def mult_head_rotary_attention(self, feature, d_out, name, num_heads, neigh_idx, is_training, xyz, nn):  # My improvement 2
        """
        Multi-head latent attention with rotary position encoding
        :param feature: Center point feature [B, N, 1, C]
        :param d_out: Output dimension
        :param name: Layer name
        :param num_heads: Number of attention heads
        :param neigh_idx: Neighbor indices [B, N, K]
        :param is_training: Training mode
        :param xyz: Coordinates [B, N, K, 3]
        :return: Aggregated feature [B, N, d_out]
        """
        # ===== 1. Parameter initialization =====
        B = tf.shape(feature)[0]
        N = tf.shape(feature)[1]
        K = tf.shape(neigh_idx)[-1]
        d_model = d_out * 2  # Latent space dimension
        head_dim = d_model // num_heads
        
        # ===== 2. Feature projection =====
        # Center point feature projection
        center_feat = tf.layers.dense(
            # tf.squeeze(feature, axis=2),  # [B, N, C]
            feature,
            d_model, 
            name=name+'_center_proj'
        )  # [B, N, d_model]
        
        # Neighbor feature projection
        nei_feat = self.random_gather(feature, neigh_idx)  # [B, N, K, C]
        dynamic_shape = tf.shape(nei_feat)  # Dynamic shape: [B, N, K, C]
        nei_feat = tf.reshape(nei_feat, [-1, nei_feat.shape[-1]])  # Static provide last dimension

        nei_feat = tf.layers.dense(nei_feat, d_model, name=name+'_nei_proj')
        nei_feat = tf.reshape(nei_feat, [dynamic_shape[0], dynamic_shape[1], dynamic_shape[2], d_model])
        # nei_feat = tf.reshape(nei_feat, [B, N, K, d_model])  # [B, N, K, d_model]
        
        # ===== 3. Rotary position encoding =====
        # Compute relative coordinates
        center_xyz = xyz[:, :, 0:1, :]  # [B, N, 1, 3]
        rel_xyz = xyz - center_xyz  # [B, N, K, 3]
        
        # Compute distance and angles (spherical coordinates)
        r = tf.norm(rel_xyz, axis=-1, keepdims=True)  # [B, N, K, 1]
        theta = tf.acos(rel_xyz[..., 2:3] / (r + 1e-6))  # Pitch angle [0, π]
        phi = tf.atan2(rel_xyz[..., 1:2], rel_xyz[..., 0:1])  # Azimuth angle [0, 2π]
        
        # Generate rotation matrix parameters
        freq_base = 10000.0
        inv_freq = 1.0 / (freq_base ** (tf.range(0, head_dim, 2.0) / head_dim))
        
        # Azimuth rotation matrix
        phi_emb = phi * inv_freq[None, None, None, :]  # [B, N, K, dim/2]
        cos_phi = tf.cos(phi_emb)
        sin_phi = tf.sin(phi_emb)
        
        # Pitch rotation matrix
        theta_emb = theta * inv_freq[None, None, None, :]  # [B, N, K, dim/2]
        cos_theta = tf.cos(theta_emb)
        sin_theta = tf.sin(theta_emb)
        
        # ===== 4. Apply rotary position encoding =====
        # Split features into real and imaginary parts (for rotation)
        def rotate_half(x):
            x1, x2 = tf.split(x, 2, axis=-1)
            return tf.concat([-x2, x1], axis=-1)
        
        # Apply rotation to center point feature (query)
        center_feat_reshaped = tf.reshape(center_feat, [B, N, num_heads, head_dim])
        center_real, center_imag = tf.split(center_feat_reshaped, 2, axis=-1)
        
        # Azimuth rotation   Issue with multiplication with trigonometric functions, but these functions are not defined earlier
        center_rot_phi = center_real * cos_phi[..., None, :] + rotate_half(center_real) * sin_phi[..., None, :]
        # Pitch rotation
        center_rot = center_rot_phi * cos_theta[..., None, :] + rotate_half(center_rot_phi) * sin_theta[..., None, :]
        center_rot = tf.reshape(center_rot, [B, N, num_heads, head_dim])
        
        # Apply rotation to neighbor feature (key)
        nei_feat_reshaped = tf.reshape(nei_feat, [B, N, K, num_heads, head_dim])
        nei_real, nei_imag = tf.split(nei_feat_reshaped, 2, axis=-1)
        
        # Azimuth rotation
        nei_rot_phi = nei_real * cos_phi[..., None, :] + rotate_half(nei_real) * sin_phi[..., None, :]
        # Pitch rotation
        nei_rot = nei_rot_phi * cos_theta[..., None, :] + rotate_half(nei_rot_phi) * sin_theta[..., None, :]
        nei_rot = tf.reshape(nei_rot, [B, N, K, num_heads, head_dim])
        
        # ===== 5. Multi-head latent attention =====
        # Latent space projection
        latent_q = tf.layers.dense(center_rot, head_dim, name=name+'_latent_q')  # [B, N, H, D_h]
        latent_k = tf.layers.dense(nei_rot, head_dim, name=name+'_latent_k')    # [B, N, K, H, D_h]
        
        # Attention score
        attn_scores = tf.einsum('bnhd,bnkhd->bnhk', latent_q, latent_k)  # [B, N, H, K]
        
        # Distance decay (optional)
        dist_decay = tf.exp(-r[..., 0] / 5.0)  # [B, N, K]
        attn_scores += tf.log(dist_decay)[:, :, None, :]  # Add in log space
        
        # Attention weights
        attn_weights = tf.nn.softmax(
            attn_scores / tf.sqrt(tf.cast(head_dim, tf.float32)),
            axis=-1
        )  # [B, N, H, K]
        
        # ===== 6. Feature aggregation =====
        # Weighted sum
        weighted_feat = tf.einsum('bnhk,bnkhd->bnhd', attn_weights, nei_rot)  # [B, N, H, D_h]
        
        # Merge multi-head
        output = tf.reshape(weighted_feat, [B, N, d_model])  # [B, N, d_model]
        
        # ===== 7. Output layer =====
        output = tf.layers.dense(output, d_out, name=name+'_out')  # [B, N, d_out]
        output = self.GLUE(tf.layers.batch_normalization(
            output, momentum=0.99, epsilon=1e-6, training=is_training
        ))
        return output    

    def mlp(self, feature, d_out, name, is_training):  

        feature = tf.expand_dims(feature, [2])
        feature = tf_util.conv2d(feature, d_out, [1, 1], name, [1, 1], 'VALID', True, is_training)
        feature = tf.squeeze(feature, [2])

        return feature


    def GLUE(self, pc):

        weight = 0.5 * (1.0 * tf.erf(pc / tf.sqrt(2.0)))

        return pc * weight

    def inference(self, inputs, is_training):

        f_encode = []
        f_decode = []


        d_out = self.config.d_out

        feature = tf.concat([inputs['xyz'][0] / 50, inputs['features'][:, :, 3:]], axis=-1)  # Normalize point coordinates and concatenate with feature channels, then send through a fully connected layer with batch normalization + GLUE module processing

        feature = tf.layers.dense(feature, d_out[0], activation=None, name='fc0')
        feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))

        num_head = self.config.num_head
        batch = tf.shape(feature)[0]
        layer = self.config.num_layers
        volex = [1, 1, 1, 1, 1]


        for i in range(layer):   # Encoder stage

            feature_shout = feature    # Multi-heatmap attention



            xyz = self.gather_neighbour(inputs['xyz'][i], inputs['neigh_idx'][i])
            feature0 = self.mult_latent_att(feature, d_out[i] // 2, 'layers' + str(i), num_head, inputs['neigh_idx'][i], is_training, xyz, self.config.k_n * 2 - 1)
            xyz = self.gather_neighbour(inputs['xyz'][i], inputs['high'][i])
            feature1 = self.mult_latent_att(feature, d_out[i] // 2, 'layers0' + str(i), num_head, inputs['high'][i], is_training, xyz, self.config.k_n * 2 - 1)


            feature = tf.concat([feature0, feature1], axis=-1)  # Residual connection + feature fusion
            
            feature = self.mlp(feature, d_out[i], 'mlp_f01' + str(i), is_training)  # Another MLP layer

            feature = feature + feature_shout
            feature = tf.nn.leaky_relu(feature)

            f_encode.append(feature)




            if i < layer - 1:   # Downsampling   Fuse col (color/geometry) features, and apply MLP again

                feature = self.random_sample_1(feature, inputs['sub_idx'][i])  # New random sampling layer

                col = tf.concat([inputs['xyz'][i + 1] / 50, inputs['col'][i + 1]], axis=-1)
                col = tf.layers.dense(col, d_out[i], activation=None, name='fclayer_i' + str(i))
                col = self.GLUE(tf.layers.batch_normalization(col, -1, 0.99, 1e-6, training=is_training))
            
                feature = tf.concat([col, feature], axis=-1)

                feature = self.mlp(feature, d_out[i+1], 'mlp_f02' + str(i), is_training)


        for i in range(layer - 1):  # Decoder

            feature = self.nearest_interpolation_1(feature, inputs['interp_idx'][layer - 2 - i])   # Upsampling with skip connection
            feature = self.random_sample_1(feature, inputs['neigh_idx'][layer - 2 - i])  # New interpolation (upsampling), random sampling and MLP method

            feature = tf.concat([feature, f_encode[layer - 2 - i]], axis=-1)

            feature = self.mlp(feature, d_out[layer - 2 - i], 'def01' + str(i), is_training)

            feature = feature + f_encode[layer - 2 - i]
            feature = tf.nn.leaky_relu(feature)

            if i > layer - 5:
                f_decode.append(feature)



        feature = self.mlp(feature, 64, 'class0', is_training)  # Classification head   MLP + fully connected layer for classification output, originally only used tf_util.conv2d
        feature = self.mlp(feature, 32, 'class1', is_training)
        feature = tf.layers.dense(feature, self.config.num_classes, activation=None, name='fc30')

        return feature   # feature , multi_scale_features, f_encode, f_decode


    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        acc_list = []
        l_out_list = []
        time_list = []
        num_list = []

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy,
                       self.inputs['input_inds'],
                       self.inputs['cloud_inds'],
                       self.prob_logits]

                _, _, summary, l_out, probs, labels, acc, point_idx, cloud_idx, prob_logits = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                acc_list.append(acc)
                l_out_list.append(l_out)
                time_list.append(t_end - t_start)



                if self.training_step % 100 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                    acc_mean = np.mean(acc_list)
                    l_out_mean = np.mean(l_out_list)
                    time_mean = np.mean(time_list)
                    log_out(message.format(self.training_step, l_out_mean, acc_mean * 100, 1000 * time_mean), self.Log_file)
                    acc_list = []
                    l_out_list = []
                    time_list = []
                self.training_step += 1

            except tf.errors.OutOfRangeError:
                if dataset.use_val and self.training_epoch % 2 == 0:
                    m_iou = self.evaluate(dataset)
                    if m_iou > np.max(self.mIou_list):
                        # Save the best model
                        snapshot_directory = join(self.saving_path, 'snapshots')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                    self.mIou_list.append(m_iou)
                    log_out('Best m_IoU of {} is: {:5.3f}'.format(dataset.name, max(self.mIou_list)), self.Log_file)
                else:
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', self.training_step)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                acc_list = []
                l_out_list = []
                time_list = []
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        conf = np.zeros([self.config.num_classes, self.config.num_classes])
        val_total_correct = 0
        val_total_seen = 0
        time_list = []

        for step_id in range(self.config.val_steps):
            t_start = time.time()
            if step_id % 100 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy, self.inputs['input_inds'], self.inputs['cloud_inds'])
                stacked_prob, labels, acc, point_idx, cloud_idx = self.sess.run(ops, {self.is_training: False})
                t_end = time.time()
                time_list.append(t_end - t_start)

                labels = np.array(labels)
                pred = np.argmax(stacked_prob, 1)

                self.config.ignored_label_inds=[]
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.config.num_classes, 1))
                conf += np.array(conf_matrix)
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        print('-------time--------', int(np.mean(time_list) * 1000))
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        conf = np.array(conf) / 1000
        for i in range(conf.shape[0]):
            s = '{:5.2f}    '.format(conf[i, 0])
            for j in range(conf.shape[1] - 1):
                s += '{:5.2f}    '.format(conf[i, j + 1])
            log_out(s, self.Log_file)
        return mean_iou


    def get_loss_improved(self, logits, labels, pre_cal_weights):
        """
        Improved weighted cross-entropy loss function
        Combines label smoothing, adaptive Focal Loss, and dynamic hard example weighting
        
        Parameters:
            logits: Model output [N, C]
            labels: Ground truth labels [N]
            pre_cal_weights: Precomputed class weights [C]
        
        Returns:
            loss: Scalar loss value
        """
        # === 1. Type safety handling ===
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)
        
        # === 2. Class weight handling ===
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes, dtype=tf.float32)
        
        # Basic class weights (handle class imbalance)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        
        # === 3. Dynamic hard example weighting ===
        # Compute prediction confidence
        probs = tf.nn.softmax(logits, axis=-1)
        max_probs = tf.reduce_max(probs, axis=1)
        
        # Identify hard examples (confidence < 0.7)
        hard_sample_mask = max_probs < 0.7
        
        # Assign higher weight to hard examples (2x)
        weights = tf.where(
            hard_sample_mask, 
            weights * 2.0,  # Hard example weight doubled
            weights          # Normal examples keep original weight
        )
        
        # === 4. Label smoothing technique ===
        label_smoothing = 0.1  # Smoothing factor
        smoothed_labels = one_hot_labels * (1.0 - label_smoothing) + label_smoothing / self.config.num_classes
        
        # === 5. Cross-entropy loss calculation ===
        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, 
            labels=smoothed_labels
        )
        
        # === 6. Adaptive Focal weighting ===
        # Compute Focal factor (γ=2.0)
        focal_factor = tf.pow(1.0 - max_probs, 2.0)
        
        # Combine Focal factor and class weights
        weighted_loss = (1.0 + focal_factor) * ce_loss * weights
        
        # === 7. Loss normalization ===
        return tf.reduce_mean(weighted_loss)

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature



    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2)
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features

    @staticmethod
    def random_sample_1(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        # pool_features = pool_features[-1]
        return pool_features

    @staticmethod
    def random_sample_2(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        pool_features = tf.squeeze(pool_features, [2])
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features

    @staticmethod
    def random_gather(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features


    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def nearest_interpolation_1(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        # interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features



    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, -1, d])

        return features

