import 'package:flutter/foundation.dart';

class CaseProvider extends ChangeNotifier {
  CaseProvider();

  String? _activeCaseId;

  String? get activeCaseId => _activeCaseId;

  void setActiveCase(String caseId) {
    _activeCaseId = caseId;
    notifyListeners();
  }
}

