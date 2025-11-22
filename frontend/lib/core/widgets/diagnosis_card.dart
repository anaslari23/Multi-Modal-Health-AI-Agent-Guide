import 'package:flutter/material.dart';

class DiagnosisCard extends StatelessWidget {
  const DiagnosisCard({
    super.key,
    required this.title,
    required this.subtitle,
    this.riskLevel,
  });

  final String title;
  final String subtitle;
  final String? riskLevel;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(subtitle),
            if (riskLevel != null) ...[
              const SizedBox(height: 8),
              Chip(
                label: Text('Risk: $riskLevel'),
                backgroundColor: colorScheme.secondary.withOpacity(0.1),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
