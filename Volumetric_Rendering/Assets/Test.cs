using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Test : MonoBehaviour
{
    private void OnDrawGizmos()
    {
        Gizmos.DrawRay(transform.position, transform.forward * 5.0f);
    }
}
