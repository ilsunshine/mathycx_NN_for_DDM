labels = hdf5dataset.get_labels(subdomain_name, tho_batch).to(self.device)

# print("测试节点： labels:",labels)
# print("测试节点 labels的形状为",labels.shape)
outputs = self.network(subdomain_data, global_dataset, tho_batch)
# print("test point outputs is",outputs)
if self.prediction_model == "discrete":
    labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                                 feature_dim=self.output_dim).to(self.device)
    # test_metris.update(outputs, labels)
elif self.prediction_model == "continue":
    # print("测试节点 除以m_l前的labels' shape is ",labels.shape)
    # print("测试节点 除以m_l前的labels' values is ",labels)
    # print("测试节点 m_l's shape is ",subdomain_data['m_l'].shape)
    # print("测试节点 m_l's values is ",subdomain_data['m_l'])
    dis_labels = get_discrete_labels(labels=labels, ml=subdomain_data['m_l'],
                                     feature_dim=self.output_dim).to(self.device)
    labels = labels / (subdomain_data['m_l'].unsqueeze(-1))
    with torch.no_grad():
        # dis_outputs=outputs*(subdomain_data['m_l'].unsqueeze(-1))
        # print("-------------正确性测试 此时dis_outputs=labes,如若程序正确无误应当具有高准确性-----------")
        # dis_outputs = labels * (subdomain_data['m_l'].unsqueeze(-1))
        dis_outputs = get_discrete_labels_one_hot(labels=outputs, ml=subdomain_data['m_l'],
                                                  feature_dim=self.output_dim).to(self.device)
        # print("测试节点 dislabels is ",dis_labels)
        # print("测试节点 dis_outputs is ",dis_outputs)
        # print("测试节点 outputs is ",outputs)
        test_metris.update(dis_outputs, dis_labels)